#include "r2d2/tracker_color.hpp"
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <filesystem>
#include <ament_index_cpp/get_package_share_directory.hpp>

using namespace std::placeholders;
namespace fs = std::filesystem;

TrackerColor::TrackerColor() : Node("tracker_color"), 
                               _model_trained(false),
                               _is_tracking(false), 
                               _frame_count(0), 
                               _initial_obj_size(0.0),
                               _object_is_moving(false), 
                               _target_reached(false)
{
  // Initialize feature detector
  try {
    _detector = cv::SIFT::create(500);
    RCLCPP_INFO(get_logger(), "Using SIFT feature detector");
  } catch (...) {
    _detector = cv::ORB::create(500);
    RCLCPP_INFO(get_logger(), "Using ORB feature detector");
  }
  
  // Initialize matcher
  _matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
  
  // Load training images
  std::string training_folder = this->declare_parameter("training_folder", "");
  if (training_folder.empty()) {
    try {
      std::string package_path = ament_index_cpp::get_package_share_directory("r2d2");
      training_folder = package_path + "/training_images";
      RCLCPP_INFO(get_logger(), "Looking for training images in: %s", training_folder.c_str());
    } catch (...) {
      RCLCPP_WARN(get_logger(), "Could not find package path.");
    }
  }
  
  if (!training_folder.empty() && _loadTrainingImages(training_folder)) {
    _extractFeatures();
    _model_trained = true;
    RCLCPP_INFO(get_logger(), "Model trained - using COLOR + SHAPE detection");
  } else {
    RCLCPP_WARN(get_logger(), "No training data - using COLOR ONLY detection");
  }
  
  // HSV ranges for RED color (red wraps around in HSV, so we need 2 ranges)
  // Lower red range: 0-10
  _lower_hsv1 = cv::Scalar(0, 100, 100);
  _upper_hsv1 = cv::Scalar(10, 255, 255);
  
  // Upper red range: 170-180
  _lower_hsv2 = cv::Scalar(170, 100, 100);
  _upper_hsv2 = cv::Scalar(180, 255, 255);
  
  // Allow customization via parameters
  std::vector<long> lower1 = this->declare_parameter("lower_hsv1", std::vector<long>{0, 100, 100});
  std::vector<long> upper1 = this->declare_parameter("upper_hsv1", std::vector<long>{10, 255, 255});
  std::vector<long> lower2 = this->declare_parameter("lower_hsv2", std::vector<long>{170, 100, 100});
  std::vector<long> upper2 = this->declare_parameter("upper_hsv2", std::vector<long>{180, 255, 255});
  
  if (lower1.size() == 3) _lower_hsv1 = cv::Scalar(lower1[0], lower1[1], lower1[2]);
  if (upper1.size() == 3) _upper_hsv1 = cv::Scalar(upper1[0], upper1[1], upper1[2]);
  if (lower2.size() == 3) _lower_hsv2 = cv::Scalar(lower2[0], lower2[1], lower2[2]);
  if (upper2.size() == 3) _upper_hsv2 = cv::Scalar(upper2[0], upper2[1], upper2[2]);
  
  RCLCPP_INFO(get_logger(), "Color Detection Tracker initialized");
  RCLCPP_INFO(get_logger(), "HSV Range 1: [%.0f,%.0f,%.0f] - [%.0f,%.0f,%.0f]", 
              _lower_hsv1[0], _lower_hsv1[1], _lower_hsv1[2],
              _upper_hsv1[0], _upper_hsv1[1], _upper_hsv1[2]);
  RCLCPP_INFO(get_logger(), "HSV Range 2: [%.0f,%.0f,%.0f] - [%.0f,%.0f,%.0f]", 
              _lower_hsv2[0], _lower_hsv2[1], _lower_hsv2[2],
              _upper_hsv2[0], _upper_hsv2[1], _upper_hsv2[2]);
  
  // Subscribers
  _img_sub = create_subscription<sensor_msgs::msg::Image>(
    "/image", rclcpp::SensorDataQoS(), 
    std::bind(&TrackerColor::_imageCallback, this, _1));

  // Publishers
  _visualization_pub = create_publisher<sensor_msgs::msg::Image>(
    "/visualization", rclcpp::SensorDataQoS());
  _vel_pub = create_publisher<geometry_msgs::msg::Twist>(
    "/cmd_vel_tracker", rclcpp::SystemDefaultsQoS());

  RCLCPP_INFO(get_logger(), "TrackerColor Node started!");
}

bool TrackerColor::_loadTrainingImages(const std::string& folder_path)
{
  if (!fs::exists(folder_path) || !fs::is_directory(folder_path)) {
    RCLCPP_ERROR(get_logger(), "Training folder does not exist: %s", folder_path.c_str());
    return false;
  }
  
  std::vector<std::string> image_extensions = {".jpg", ".jpeg", ".png", ".bmp"};
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
        }
      }
    }
  }
  
  RCLCPP_INFO(get_logger(), "Loaded %d training images", loaded);
  return loaded > 0;
}

void TrackerColor::_extractFeatures()
{
  std::vector<cv::Mat> descriptors_list;
  
  for (const auto& img : _training_images) {
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    
    cv::Mat gray;
    if (img.channels() == 3) {
      cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    } else {
      gray = img;
    }
    
    _detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
    
    if (!descriptors.empty()) {
      descriptors_list.push_back(descriptors);
    }
  }
  
  if (!descriptors_list.empty()) {
    cv::vconcat(descriptors_list, _training_descriptors);
    if (_training_descriptors.type() != CV_32F) {
      _training_descriptors.convertTo(_training_descriptors, CV_32F);
    }
    RCLCPP_INFO(get_logger(), "Extracted features from %zu training images", _training_images.size());
  }
}

bool TrackerColor::_verifyObjectWithFeatures(const cv::Mat& frame, const cv::Rect& roi)
{
  if (!_model_trained || _training_descriptors.empty()) {
    return true;  // If no model, accept color detection
  }
  
  // Extract region of interest
  if (roi.x < 0 || roi.y < 0 || 
      roi.x + roi.width > frame.cols || roi.y + roi.height > frame.rows) {
    return false;
  }
  
  cv::Mat roi_img = frame(roi);
  cv::Mat gray;
  if (roi_img.channels() == 3) {
    cv::cvtColor(roi_img, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = roi_img;
  }
  
  // Detect features in ROI
  std::vector<cv::KeyPoint> keypoints;
  cv::Mat descriptors;
  _detector->detectAndCompute(gray, cv::noArray(), keypoints, descriptors);
  
  if (descriptors.empty() || keypoints.size() < MIN_MATCH_COUNT) {
    return false;
  }
  
  if (descriptors.type() != CV_32F) {
    descriptors.convertTo(descriptors, CV_32F);
  }
  
  // Match features
  std::vector<std::vector<cv::DMatch>> knn_matches;
  try {
    _matcher->knnMatch(descriptors, _training_descriptors, knn_matches, 2);
  } catch (const cv::Exception& e) {
    RCLCPP_WARN(get_logger(), "Feature matching failed: %s", e.what());
    return false;
  }
  
  // Apply ratio test
  std::vector<cv::DMatch> good_matches;
  for (size_t i = 0; i < knn_matches.size(); i++) {
    if (knn_matches[i].size() >= 2 && 
        knn_matches[i][0].distance < MATCH_RATIO_THRESH * knn_matches[i][1].distance) {
      good_matches.push_back(knn_matches[i][0]);
    }
  }
  
  bool verified = good_matches.size() >= MIN_MATCH_COUNT;
  
  RCLCPP_INFO(get_logger(), "Feature verification: %zu matches (min: %d) - %s", 
              good_matches.size(), MIN_MATCH_COUNT, verified ? "VERIFIED" : "REJECTED");
  
  return verified;
}

cv::Rect TrackerColor::_detectColoredObject(const cv::Mat& frame)
{
  // Convert to HSV
  cv::Mat hsv;
  cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
  
  // Create masks for both red ranges
  cv::Mat mask1, mask2, mask;
  cv::inRange(hsv, _lower_hsv1, _upper_hsv1, mask1);
  cv::inRange(hsv, _lower_hsv2, _upper_hsv2, mask2);
  
  // Combine masks
  cv::bitwise_or(mask1, mask2, mask);
  
  // Count non-zero pixels in mask
  int red_pixels = cv::countNonZero(mask);
  RCLCPP_DEBUG(get_logger(), "Red pixels found: %d", red_pixels);
  
  // Morphological operations to remove noise
  cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
  cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
  cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
  
  // Find contours
  std::vector<std::vector<cv::Point>> contours;
  cv::findContours(mask, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
  
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000, 
                       "Color detection: %d red pixels, %zu contours found", 
                       red_pixels, contours.size());
  
  if (contours.empty()) {
    return cv::Rect();
  }
  
  // Find the largest contour that looks circular and passes verification
  cv::Rect best_rect;
  std::vector<std::pair<cv::Rect, double>> candidates;  // Store all valid candidates
  
  int rejected_by_area = 0;
  int rejected_by_circularity = 0;
  
  for (const auto& contour : contours) {
    double area = cv::contourArea(contour);
    
    // Filter by area - ignore small textures
    if (area < MIN_CONTOUR_AREA) {
      rejected_by_area++;
      continue;
    }
    
    if (area > MAX_CONTOUR_AREA) {
      rejected_by_area++;
      continue;
    }
    
    // Calculate circularity
    double perimeter = cv::arcLength(contour, true);
    if (perimeter == 0) continue;
    
    double circularity = 4.0 * CV_PI * area / (perimeter * perimeter);
    double score = area * circularity;  // Prefer larger, more circular objects
    
    RCLCPP_DEBUG(get_logger(), "Contour: area=%.0f, circ=%.2f", area, circularity);
    
    if (circularity > MIN_CIRCULARITY) {
      cv::Rect candidate_rect = cv::boundingRect(contour);
      candidates.push_back({candidate_rect, score});
      
      // Store circularity for later use
      RCLCPP_DEBUG(get_logger(), "Added candidate: circularity=%.2f", circularity);
    } else {
      rejected_by_circularity++;
    }
  }
  
  RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                       "Filtering: %d rejected by area, %d by circularity, %zu candidates remaining", 
                       rejected_by_area, rejected_by_circularity, candidates.size());
  
  // Sort candidates by score (largest first)
  std::sort(candidates.begin(), candidates.end(), 
            [](const auto& a, const auto& b) { return a.second > b.second; });
  
  // Try candidates from largest to smallest
  for (const auto& [candidate_rect, score] : candidates) {
    RCLCPP_INFO(get_logger(), "Checking candidate at (%d,%d) size %dx%d, score=%.0f", 
                candidate_rect.x, candidate_rect.y, 
                candidate_rect.width, candidate_rect.height, score);
    
    // For small circular objects, features might not work well
    // Accept if very circular (score is area * circularity, high score = good circle)
    bool accept_without_features = (score > 100);  // Lower threshold for distant objects
    
    // Verify with feature matching if model is trained
    if (_model_trained && !accept_without_features) {
      if (_verifyObjectWithFeatures(frame, candidate_rect)) {
        best_rect = candidate_rect;
        RCLCPP_INFO(get_logger(), "Object VERIFIED with features!");
        break;  // Found it!
      }
    } else {
      // Accept based on color/shape only (no model OR high circularity score)
      best_rect = candidate_rect;
      if (accept_without_features) {
        RCLCPP_INFO(get_logger(), "Object accepted (high circularity score: %.0f)", score);
      } else {
        RCLCPP_INFO(get_logger(), "Object accepted (no model)");
      }
      break;
    }
  }
  
  if (best_rect.width > 0 && best_rect.height > 0) {
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 2000,
                         "Detected verified object at (%d,%d) size %dx%d", 
                         best_rect.x, best_rect.y, best_rect.width, best_rect.height);
    return best_rect;
  }
  
  return cv::Rect();
}

void TrackerColor::_trackAndFollow(cv::Mat& frame, geometry_msgs::msg::Twist& vel_msg, 
                                   uint32_t img_width, uint32_t img_height)
{
  if (!_is_tracking) {
    // Try to detect object by color
    cv::Rect detected = _detectColoredObject(frame);
    
    if (detected.width > 0 && detected.height > 0) {
      // Object found! Initialize tracker
      _tracker = cv::TrackerCSRT::create();
      _tracker->init(frame, detected);
      _is_tracking = true;
      _frame_count = 0;
      _initial_obj_size = (detected.width * detected.height) / 
                          static_cast<double>(img_width * img_height);
      
      RCLCPP_INFO(get_logger(), "RED object detected! Starting tracking...");
      
      // Draw detection
      cv::rectangle(frame, detected, cv::Scalar(0, 255, 0), 3);
      cv::putText(frame, "RED Object Detected!", cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);
    } else {
      // Object not detected, rotate to search
      vel_msg.angular.z = SEARCH_ANG_VEL;
      cv::putText(frame, "Searching for RED object...", cv::Point(50, 50), 
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
      
      // Track object center position for motion detection
      cv::Point obj_center(tracked_box.x + tracked_box.width / 2, 
                           tracked_box.y + tracked_box.height / 2);
      _object_positions.push_back(obj_center);
      
      // Keep only last 10 positions
      if (_object_positions.size() > 10) {
        _object_positions.erase(_object_positions.begin());
      }
      
      // Check if object is moving (compare first and last position in buffer)
      _object_is_moving = false;
      if (_object_positions.size() >= 5) {
        cv::Point first = _object_positions.front();
        cv::Point last = _object_positions.back();
        double distance = cv::norm(first - last);
        
        // If moved more than 30 pixels in last 5 frames, it's moving
        if (distance > 30.0) {
          _object_is_moving = true;
          _target_reached = false;  // Reset target reached if object moves
          RCLCPP_DEBUG(get_logger(), "Object moving: distance %.1f pixels", distance);
        } else {
          RCLCPP_DEBUG(get_logger(), "Object stationary: distance %.1f pixels", distance);
        }
      }
      
      // Check if we reached the target distance
      if (obj_size >= TARGET_OBJECT_SIZE) {
        if (!_target_reached) {
          RCLCPP_INFO(get_logger(), "Target distance reached! Object size: %.3f", obj_size);
        }
        _target_reached = true;
      }
      
      // Periodically verify detection with color (every 20 frames)
      _frame_count++;
      if (_frame_count % 20 == 0) {
        cv::Rect verified = _detectColoredObject(frame);
        if (verified.width > 0 && verified.height > 0) {
          // Re-initialize with detected position
          _tracker = cv::TrackerCSRT::create();
          _tracker->init(frame, verified);
          tracked_box = verified;
          RCLCPP_DEBUG(get_logger(), "Tracker re-initialized with color detection");
        }
      }
      
      // Control robot to follow
      _designateControl(vel_msg, tracked_box, img_width, img_height);
      
      // If target reached and object is not moving, stop the robot
      if (_target_reached && !_object_is_moving) {
        vel_msg.linear.x = 0.0;
        vel_msg.angular.z = 0.0;
        RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 1000, 
                             "Robot STOPPED - target reached and object stationary");
      }
      
      int obj_x_center = tracked_box.x + tracked_box.width / 2;
      
      // Visualize
      cv::rectangle(frame, tracked_box, cv::Scalar(255, 0, 0), 2);
      cv::line(frame, cv::Point(img_width/2, 0), 
               cv::Point(img_width/2, img_height), cv::Scalar(0, 255, 0), 1);
      cv::circle(frame, cv::Point(obj_x_center, tracked_box.y + tracked_box.height/2), 
                 5, cv::Scalar(0, 0, 255), -1);
      
      // Status message
      std::string status;
      if (_target_reached && !_object_is_moving) {
        status = "Target Reached - Waiting";
      } else if (_target_reached && _object_is_moving) {
        status = "Following (target moving)";
      } else if (_object_is_moving) {
        status = "Following (moving target)";
      } else {
        status = "Following (stationary)";
      }
      
      cv::putText(frame, status, cv::Point(50, 50), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.75, 
                  (_target_reached && !_object_is_moving) ? cv::Scalar(0, 255, 255) : cv::Scalar(0, 255, 0), 2);
      _target_reached = false;
      _object_positions.clear();
      
      char info[200];
      snprintf(info, sizeof(info), "V:%.2f A:%.2f Size:%.1f%% [%dx%d] %s", 
               vel_msg.linear.x, vel_msg.angular.z, obj_size * 100, 
               tracked_box.width, tracked_box.height,
               _object_is_moving ? "MOVING" : "STILL");
      cv::putText(frame, info, cv::Point(50, 80), 
                  cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
      
    } else {
      // Tracking lost, go back to detection mode
      RCLCPP_WARN(get_logger(), "Tracking lost. Switching back to color detection mode.");
      _is_tracking = false;
      vel_msg.angular.z = SEARCH_ANG_VEL;
    }
  }
}

void TrackerColor::_imageCallback(const sensor_msgs::msg::Image::SharedPtr msg)
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
  _trackAndFollow(frame, vel_msg, msg->width, msg->height);
  
  // Publish velocity
  _vel_pub->publish(vel_msg);
  
  // Publish visualization
  cv_image->image = frame;
  auto img_msg = cv_image->toImageMsg();
  _visualization_pub->publish(*img_msg);
  
  // Show locally
  cv::imshow("TrackerColor", frame);
  cv::waitKey(1);
}

void TrackerColor::_designateControl(geometry_msgs::msg::Twist &vel_msg, cv::Rect obj, 
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
  auto node = std::make_shared<TrackerColor>();
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
