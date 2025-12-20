#pragma once

#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/twist.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <opencv2/tracking.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>

class TrackerML : public rclcpp::Node
{
public:
  constexpr static float MIN_ANG_VEL = 0.15f;
  constexpr static float MAX_ANG_VEL = 0.5f;
  constexpr static float ANGULAR_GAIN = 1.7f;
  constexpr static float SEARCH_ANG_VEL = 0.3f;
  constexpr static double MAX_LINEAR_VEL = 0.5;
  constexpr static double MIN_LINEAR_VEL = 0.1;
  constexpr static double DISTANCE_GAIN = 2.0;
  constexpr static double TARGET_OBJECT_SIZE = 0.3;
  constexpr static int MIN_MATCH_COUNT = 10;

  TrackerML();

private:
  void _imageCallback(const sensor_msgs::msg::Image::SharedPtr msg);
  bool _loadTrainingImages(const std::string& folder_path);
  void _extractFeatures();
  cv::Rect _detectObject(const cv::Mat& frame);
  void _designateControl(geometry_msgs::msg::Twist &vel_msg, cv::Rect obj, uint32_t img_width, uint32_t img_height);
  void _trackAndFollow(cv::Mat& frame, geometry_msgs::msg::Twist& vel_msg, uint32_t img_width, uint32_t img_height);

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr _img_sub;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr _visualization_pub;
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr _vel_pub;
  
  // ML/Feature detection members
  cv::Ptr<cv::Feature2D> _detector;
  cv::Ptr<cv::DescriptorMatcher> _matcher;
  std::vector<cv::Mat> _training_images;
  std::vector<cv::KeyPoint> _training_keypoints;
  cv::Mat _training_descriptors;
  
  // Tracking members
  cv::Ptr<cv::Tracker> _tracker;
  bool _is_tracking;
  bool _model_trained;
  int _frame_count;
  double _initial_obj_size;
};
