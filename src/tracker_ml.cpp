#include "r2d2/tracker_ml.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace std::placeholders;
namespace fs = std::filesystem;

TrackerML::TrackerML() : Node("tracker_ml"), _is_tracking(false), _model_trained(false), 
                         _frame_count(0), _initial_obj_size(0.0)
{
  // Initialize feature detector (ORB is free and fast, SIFT is more robust but patented)
  // Try to use SIFT if available, otherwise fall back to ORB
  try {
    _detector = cv::SIFT::create(1000);  // Create with 1000 features
    RCLCPP_INFO(get_logger(), "Using SIFT feature detector");
  } catch (...) {
    _detector = cv::ORB::create(1000);
    RCLCPP_INFO(get_logger(), "Using ORB feature detector (SIFT not available)");
  }
  
  // Initialize matcher
  _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  
  // Load training images from the training folder
  std::string training_folder = this->declare_parameter("training_folder", "");
  
  if (training_folder.empty()) {
    // Try default location in package
    try {
      std::string package_path = ament_index_cpp::get_package_share_directory("r2d2");
      training_folder = package_path + "/training_images";
    } catch (...) {
      RCLCPP_WARN(get_logger(), "Could not find package path. Please specify training_folder parameter.");
    }
  }
  
  if (!training_folder.empty() && _loadTrainingImages(training_folder)) {
    _extractFeatures();
    _model_trained = true;
    RCLCPP_INFO(get_logger(), "Model trained on %zu images with %zu keypoints", 
                _training_images.size(), _training_keypoints.size());
  } else {
    RCLCPP_WARN(get_logger(), "No training images found. Will wait for manual ROI selection.");
    RCLCPP_WARN(get_logger(), "To use ML detection, provide images in: %s", training_folder.c_str());
  }
  
  // Subscribers
  _img_sub = create_subscription<sensor_msgs::msg::Image>(
    "/image", rclcpp::SensorDataQoS(), 
    std::bind(&TrackerML::_imageCallback, this, _1));

  // Publishers
  _visualization_pub = create_publisher<sensor_msgs::msg::Image>(
    "/visualization", rclcpp::SensorDataQoS());
  _vel_pub = create_publisher<geometry_msgs::msg::Twist>(
    "/cmd_vel_tracker", rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "TrackerML Node started!");
}

bool TrackerML::_loadTrainingImages(const std::string& folder_path)
{
  if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
    RCLCPP_ERROR(get_logger(), "Training folder does not exist: %s", folder_path.c_str());
    return false;
  }
  
  std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"};
  int loaded = 0;
  
  for (const auto& entry : fs::directory_iterator(folder_path)) {
    if (entry.is_regular_file()) {
      std::string ext = entry.path().extension().string();
      std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
      
      if (std::find(image_extensions.begin(), image_extensions.end(), ext) != image_extensions.end()) {
        cv::Mat img = cv::imread(entry.path().string());
        if (!img.empty()) {
          _training_images.push_back(img);
          loaded++;
          RCLCPP_INFO(get_logger(), "Loaded training image: %s", entry.path().filename().c_str());
        }
      }
    }
  }
  
  if (loaded == 0) {
    RCLCPP_ERROR(get_logger(), "No valid images found in: %s", folder_path.c_str());
    return false;
  }
  
  return true;
}

void TrackerML::_extractFeatures()
{
  _training_keypoints.clear();
  std::vector<cv::Mat> descriptors_list;
  
  for (const auto& img : _training_images) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    // Convert to grayscale if needed
    cv::Mat gray;
    if (img.channels() == 3) {
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = img;
    }
    
    // Detect and compute features
    _detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    if (!descriptors.empty()) {
      // Store keypoints and descriptors
      _training_keypoints.insert(_training_keypoints.end(), keypoints.begin(), keypoints.end());
      descriptors_list.push_back(descriptors);
    }
  }
  
  // Concatenate all descriptors
  if (!descriptors_list.empty()) {
    cv::vconcat(descriptors_list, _training_descriptors);
    
    // Convert descriptors to proper type for FLANN matcher if needed
    if (_training_descriptors.type() != CV_32F) {
      _training_descriptors.convertTo(_training_descriptors, CV_32F);
    }
    
    RCLCPP_INFO(get_logger(), "Extracted %d keypoints from training images", _training_keypoints.size());
  }
}

cv::Rect TrackerML::_detectObject(const cv::Mat& frame)
{
  if (_training_descriptors.empty() || !_model_trained) {
    return cv::Rect();
  }
  
  // Convert frame to grayscale
  cv::Mat gray;
  if (frame.channels() == 3) {
    cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = frame;
  }
  
  // Detect features in current frame
  std::vector<cv::KeyPoint> frame_keypoints;
  cv::Mat frame_descriptors;
  _detector->detectAndCompute(gray, cv::noArray(), frame_keypoints, frame_descriptors);
  
  if (frame_descriptors.empty() || frame_keypoints.size() < MIN_MATCH_COUNT) {
    return cv::Rect();
  }
  
  // Convert to proper type for matching
  if (frame_descriptors.type() != CV_32F) {
    frame_descriptors.convertTo(frame_descriptors, CV_32F);
  }
  
  // Match features
  std::vector<std::vector<cv::DMatch>> knn_matches;
  try {
    _matcher->knnMatch(frame_descriptors, _training_descriptors, knn_matches, 2);
  } catch (const cv::Exception& e) {
    RCLCPP_ERROR(get_logger(), "Matching failed: %s", e.what());
    return cv::Rect();
  }
  
  // Apply ratio test (Lowe's ratio test)
  std::vector<cv::DMatch> good_matches;
  const float ratio_thresh = 0.7f;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i].size() >= 2 && 
        knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  
  RCLCPP_DEBUG(get_logger(), "Found %zu good matches", good_matches.size());
  
  // Need minimum number of matches
  if (good_matches.size() < MIN_MATCH_COUNT) {
    return cv::Rect();
  }
  
  // Calculate bounding box from matched keypoints in frame
  std::vector<cv::Point2f> matched_points;
  for (const auto& match : good_matches) {
    matched_points.push_back(frame_keypoints[match.queryIdx].pt);
  }
  
  if (matched_points.empty()) {
    return cv::Rect();
  }
  
  // Find bounding rectangle
  cv::Rect bbox = cv::boundingRect(matched_points);
  
  // Add some margin (20%)
  int margin_x = bbox.width * 0.2;
  int margin_y = bbox.height * 0.2;
  bbox.x = std::max(0, bbox.x - margin_x);
  bbox.y = std::max(0, bbox.y - margin_y);
  bbox.width = std::min(frame.cols - bbox.x, bbox.width + 2 * margin_x);
  bbox.height = std::min(frame.rows - bbox.y, bbox.height + 2 * margin_y);
  
  RCLCPP_INFO(get_logger(), "Object detected at (%d, %d) size %dx%d with %zu matches", 
              bbox.x, bbox.y, bbox.width, bbox.height, good_matches.size());
  
  return bbox;
}

void TrackerML::_trackAndFollow(cv::Mat& frame, geometry_msgs::msg::Twist& vel_msg, 
                                uint32_t img_width, uint32_t img_height)
{
  if (!_is_tracking) {
    // Try to detect object
    cv::Rect detected = _detectObject(frame);
    
    if (detected.width > 0 && detected.height > 0) {
      // Object found! Initialize tracker
      _tracker = cv::TrackerCSRT::create();
      _tracker->init(frame, detected);
      _is_tracking = true;
      _frame_count = 0;
      _initial_obj_size = (detected.width * detected.height) / 
                          static_cast<double>(img_width * img_height);
      
      RCLCPP_INFO(get_logger(), "Started tracking detected object");
      
      // Draw detection
      cv::rectangle(frame, detected, cv::Scalar(0, 255, 0), 3);
      cv::putText(frame, "Object Detected!", cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    } else {
      // Object not detected, rotate to search
      vel_msg.angular.z = SEARCH_ANG_VEL;
      cv::putText(frame, "Searching for object...", cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 165, 255), 2);
    }
  } else {
    // Update tracker
    cv::Rect tracked_box;
    bool ok = _tracker->update(frame, tracked_box);
    
    if (ok) {
      // Calculate object size
      double obj_size = (tracked_box.width * tracked_box.height) / 
                        static_cast<double>(img_width * img_height);
      
      // Periodically verify detection (every 60 frames)
      _frame_count++;
      if (_frame_count % 60 == 0) {
        cv::Rect verified = _detectObject(frame);
        if (verified.width > 0 && verified.height > 0) {
          // Re-initialize with detected position
          _tracker = cv::TrackerCSRT::create();
          _tracker->init(frame, verified);
          tracked_box = verified;
          RCLCPP_DEBUG(get_logger(), "Tracker re-initialized with detection");
        }
      }
      
      // Control robot to follow
      _designateControl(vel_msg, tracked_box, img_width, img_height);
      
      int obj_x_center = tracked_box.x + tracked_box.width / 2;
      
      // Visualize
      cv::rectangle(frame, tracked_box, cv::Scalar(255, 0, 0), 2);
      cv::line(frame, cv::Point(img_width/2, 0), 
               cv::Point(img_width/2, img_height), cv::Scalar(0, 255, 0), 1);
      cv::circle(frame, cv::Point(obj_x_center, tracked_box.y + tracked_box.height/2), 
                 5, cv::Scalar(0, 0, 255), -1);
      
      std::string status = (obj_size >= TARGET_OBJECT_SIZE) ? "Target reached" : "Following";
      cv::putText(frame, status, cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, cv::Scalar(0, 255, 0), 2);
      
      char info[150];
      snprintf(info, sizeof(info), "V:%.2f A:%.2f Size:%.1f%% [%dx%d]", 
               vel_msg.linear.x, vel_msg.angular.z, obj_size * 100, 
               tracked_box.width, tracked_box.height);
      cv::putText(frame, info, cv::Point(50, 80), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      
    } else {
      // Tracking lost, go back to detection mode
      RCLCPP_WARN(get_logger(), "Tracking lost. Switching back to detection mode.");
      _is_tracking = false;
      vel_msg.angular.z = SEARCH_ANG_VEL;
    }
  }
}

void TrackerML::_imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
{
  // Convert the image message to an OpenCV Mat
  cv_bridge::CvImagePtr cv_image;
  try {
    cv_image = cv_bridge::toCvCopy(msg, "bgr8");
  } catch (cv_bridge::Exception& e) {
    RCLCPP_ERROR(get_logger(), "cv_bridge exception: %s", e.what());
    return;
  }
  
  cv::Mat frame = cv_image->image;
  geometry_msgs::msg::Twist vel_msg;
  vel_msg.linear.x = 0.0;
  vel_msg.angular.z = 0.0;
  
  // Process frame
  if (_model_trained) {
    _trackAndFollow(frame, vel_msg, msg->width, msg->height);
  } else {
    cv::putText(frame, "No training data! Add images to training folder.", 
                cv::Point(50, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(0, 0, 255), 2);
  }
  
  // Publish velocity
  _vel_pub->publish(vel_msg);
  
  // Publish visualization
  cv_image->image = frame;
  auto img_msg = cv_image->toImageMsg();
  _visualization_pub->publish(*img_msg);
  
  // Show locally
  cv::imshow("TrackerML", frame);
  cv::waitKey(1);
}

void TrackerML::_designateControl(geometry_msgs::msg::Twist &vel_msg, cv::Rect obj, 
                                  uint32_t img_width, uint32_t img_height)
{
  // Calculate angular velocity based on horizontal position
  int obj_x_center = obj.x + obj.width / 2;
  int px_to_center = img_width / 2 - obj_x_center;
  float ang_vel = ANGULAR_GAIN * px_to_center / static_cast<float>(img_width);

  // Clamp angular velocity
  if (std::abs(ang_vel) < MIN_ANG_VEL) {
    vel_msg.angular.z = 0.0;
  } else if (ang_vel > MAX_ANG_VEL) {
    vel_msg.angular.z = MAX_ANG_VEL;
  } else if (ang_vel < -MAX_ANG_VEL) {
    vel_msg.angular.z = -MAX_ANG_VEL;
  } else {
    vel_msg.angular.z = ang_vel;
  }

  // Calculate linear velocity based on object size
  double obj_size = (obj.width * obj.height) / static_cast<double>(img_width * img_height);
  
  if (obj_size >= TARGET_OBJECT_SIZE) {
    // Target reached - stop
    vel_msg.linear.x = 0.0;
  } else {
    // Move forward based on distance
    double size_error = TARGET_OBJECT_SIZE - obj_size;
    double linear_vel = DISTANCE_GAIN * size_error;
    
    // Clamp velocity
    linear_vel = std::max(MIN_LINEAR_VEL, std::min(MAX_LINEAR_VEL, linear_vel));
    
    // Reduce speed when turning
    double turn_factor = 1.0 - std::abs(vel_msg.angular.z) / MAX_ANG_VEL;
    linear_vel *= (0.5 + 0.5 * turn_factor);
    
    vel_msg.linear.x = linear_vel;
  }
}

int main(int argc, char **argv)
{
  rclcpp::init(argc, argv);
  auto node = std::make_shared<TrackerML>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
