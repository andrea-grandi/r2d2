#include "r2d2/tracker.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>

using namespace std::placeholders;

Tracker::Tracker() : Node("tracker"), _is_tracker_initialized(false), _is_searching(false)
{
  // Subscribers
  _img_sub = create_subscription<sensor_msgs::msg::Image>("/image", rclcpp::SensorDataQoS(), bind(&Tracker::_imageCallback, this, _1));

  // Publishers
  _visualization_pub = create_publisher<sensor_msgs::msg::Image>("/visualization", rclcpp::SensorDataQoS());
  _vel_pub = create_publisher<geometry_msgs::msg::Twist>("/cmd_vel", rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "Node started!");
}

void Tracker::_imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Convert the image message to an OpenCV Mat
  cv_bridge::CvImagePtr cv_image = cv_bridge::toCvCopy(msg, "bgr8");
  cv::Mat frame = cv_image->image;
  cv::Rect obj;
  geometry_msgs::msg::Twist vel_msg;

  if (!_is_tracker_initialized)
  {
    _initTracker(frame, obj);
  }

  bool ok = _tracker->update(frame, obj);

  // Check if user wants to reinitialize tracker (press 'r' key)
  int key = cv::waitKey(1);
  if (key == 'r' || key == 'R') {
    RCLCPP_INFO(get_logger(), "Reinitializing tracker...");
    _is_tracker_initialized = false;
    return;
  }

  // Initialize velocity message to zero
  vel_msg.linear.x = 0.0;
  vel_msg.angular.z = 0.0;

  if (ok) {
    // Calculate angular speed based on the position of the object
    _designateControl(vel_msg, obj, msg->width);
    
    int obj_x_center = obj.x + obj.width / 2;
    int px_to_center = msg->width / 2 - obj_x_center;
    RCLCPP_INFO(get_logger(), "Angular velocity: %0.2f, Object center: %d, Pixels from center: %d", 
                vel_msg.angular.z, obj_x_center, px_to_center);
    
    // Publish visualization with rectangle around the tracked object
    rectangle(frame, obj, cv::Scalar(255, 0, 0), 2, 1);
    
    // Draw center line and object center for debugging
    cv::line(frame, cv::Point(msg->width/2, 0), cv::Point(msg->width/2, msg->height), cv::Scalar(0, 255, 0), 1);
    cv::circle(frame, cv::Point(obj_x_center, obj.y + obj.height/2), 5, cv::Scalar(0, 0, 255), -1);
  }
  else {
    // Tracking failed - start searching for the object
    _is_searching = true;
    RCLCPP_WARN(get_logger(), "Tracking failure detected. Searching for object...");
    putText(frame, "Searching for object...", cv::Point(50, 80), cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 165, 255), 2);
    
    // Try to find the object using template matching
    if (!_object_template.empty()) {
      cv::Rect found_obj = _searchForObject(frame);
      
      if (found_obj.width > 0 && found_obj.height > 0) {
        // Object found! Reinitialize tracker
        RCLCPP_INFO(get_logger(), "Object found! Reinitializing tracker...");
        _tracker = cv::TrackerCSRT::create();
        _tracker->init(frame, found_obj);
        _is_searching = false;
        
        rectangle(frame, found_obj, cv::Scalar(0, 255, 0), 2, 1);
      } else {
        // Object not found, rotate to search
        vel_msg.angular.z = SEARCH_ANG_VEL;
      }
    } else {
      // No template, just rotate
      vel_msg.angular.z = SEARCH_ANG_VEL;
    }
  }

  _vel_pub->publish(vel_msg);

  // Publish visualization
  cv_image->image = frame;
  auto img_msg = cv_image->toImageMsg();
  _visualization_pub->publish(*img_msg);
  
  // Show the frame locally for debugging (press 'r' to reinit, 'q' to quit)
  cv::imshow("Tracker", frame);
  cv::waitKey(1);
}

cv::Rect Tracker::_searchForObject(const cv::Mat& frame)
{
  if (_object_template.empty() || _object_template.cols <= 0 || _object_template.rows <= 0) {
    return cv::Rect();
  }
  
  // Use template matching to find the object
  cv::Mat result;
  int match_method = cv::TM_CCOEFF_NORMED;
  
  // Check if frame is large enough
  if (frame.cols < _object_template.cols || frame.rows < _object_template.rows) {
    return cv::Rect();
  }
  
  cv::matchTemplate(frame, _object_template, result, match_method);
  
  // Find the best match
  double minVal, maxVal;
  cv::Point minLoc, maxLoc;
  cv::minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
  
  // Threshold for accepting a match
  const double threshold = 0.6;
  
  if (maxVal > threshold) {
    cv::Rect found_rect(maxLoc.x, maxLoc.y, _object_template.cols, _object_template.rows);
    RCLCPP_INFO(get_logger(), "Template match score: %.2f", maxVal);
    return found_rect;
  }
  
  return cv::Rect();
}

void Tracker::_initTracker(cv::Mat frame, cv::Rect obj)
{
  obj = selectROI("ROI selector", frame, false);
  _tracker = cv::TrackerCSRT::create();
  _tracker->init(frame, obj);
  _is_tracker_initialized = true;
  _is_searching = false;
  
  // Save template of the object for later search
  if (obj.width > 0 && obj.height > 0) {
    _object_template = frame(obj).clone();
    RCLCPP_INFO(get_logger(), "Object template saved (%dx%d)", _object_template.cols, _object_template.rows);
  }
  
  cv::destroyWindow("ROI selector");
  cv::waitKey(1);
}

void Tracker::_designateControl(geometry_msgs::msg::Twist &vel_msg, cv::Rect obj, uint32_t img_width)
{
    int obj_x_center = obj.x + obj.width / 2;
    int px_to_center = img_width / 2 - obj_x_center;
    float ang_vel = ANGULAR_GAIN * px_to_center / static_cast<float>(img_width);

    // Ensure angular velocity is within bounds
    if (std::abs(ang_vel) < MIN_ANG_VEL) {
      vel_msg.angular.z = 0.0;
    }
    else if (ang_vel > MAX_ANG_VEL) {
      vel_msg.angular.z = MAX_ANG_VEL;
    }
    else if (ang_vel < -MAX_ANG_VEL) {
      vel_msg.angular.z = -MAX_ANG_VEL;
    }
    else {
      vel_msg.angular.z = ang_vel;
    }
}


int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<Tracker>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}