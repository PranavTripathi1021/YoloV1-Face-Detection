#include<opencv2/opencv.hpp>
#include<torch/script.h>
#include<vector>

torch::Tensor imgToTensor(cv::Mat& img_fromCV);

std::vector<std::vector<float>> predToBoxes(
    const torch::Tensor& pred,
    float threshold,
    int S =7,
    int C = 1,
    int B = 2
);

std::vector<int> yoloToCoordinates(
    const std::vector<float>& box,
    int img_width,
    int img_height
);

std::vector<std::vector<float>> applyNMS(
    const std::vector<std::vector<float>>& boxes,
    int img_width,
    int img_height,
    float score_threshold,
    float nms_threshold
);