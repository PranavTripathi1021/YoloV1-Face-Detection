#include<opencv2/opencv.hpp>
#include "functions.hpp"
#include<torch/script.h>
#include<opencv2/dnn.hpp>
#include<vector>
#include<algorithm>

torch::Tensor imgToTensor(cv::Mat& img_fromCV){
    cv::Mat img;
    cv::resize(img_fromCV, img, cv::Size(448, 448), cv::INTER_LINEAR);
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    std::vector<int64_t> dim = {img.rows, img.cols, img.channels()};
    torch::Tensor img_tensor = torch::from_blob(img.data, dim, torch::kUInt8);
    img_tensor = img_tensor.permute({2, 0, 1});
    img_tensor = img_tensor.to(torch::kFloat32).div(255.0);
    img_tensor = img_tensor.unsqueeze(0);
    return img_tensor.clone();
}

std::vector<std::vector<float>> predToBoxes(
    const torch::Tensor& pred,
    float threshold,
    int S,
    int C,
    int B
) {
    std::vector<std::vector<float>> res;

    auto pred_acc = pred.accessor<float, 3>();

    for (int row = 0; row < S; row++) {
        for (int col = 0; col < S; col++) {

            float class_prob = pred_acc[row][col][0];

            for (int b = 0; b < B; b++) {

                int conf_idx = 1 + b;
                int box_idx = 3 + b * 4;

                float confidence = pred_acc[row][col][conf_idx];
                float score = confidence * class_prob;

                if (score < threshold)
                    continue;

                float x = pred_acc[row][col][box_idx];
                float y = pred_acc[row][col][box_idx + 1];
                float w = pred_acc[row][col][box_idx + 2];
                float h = pred_acc[row][col][box_idx + 3];

                float global_x = (col + x) / S;
                float global_y = (row + y) / S;

                res.push_back({
                    global_x,
                    global_y,
                    w,
                    h,
                    score
                });
            }
        }
    }

    return res;
}

std::vector<int> yoloToCoordinates(
    const std::vector<float>& box,
    int img_width,
    int img_height
) {
    float x = box[0];
    float y = box[1];
    float w = box[2];
    float h = box[3];

    int x1 = static_cast<int>((x - w / 2) * img_width);
    int y1 = static_cast<int>((y - h / 2) * img_height);

    int x2 = static_cast<int>((x + w / 2) * img_width);
    int y2 = static_cast<int>((y + h / 2) * img_height);

    // Clamp coordinates to image boundaries
    x1 = std::clamp(x1, 0, img_width - 1);
    y1 = std::clamp(y1, 0, img_height - 1);

    x2 = std::clamp(x2, 0, img_width - 1);
    y2 = std::clamp(y2, 0, img_height - 1);

    return {x1, y1, x2, y2};
}

std::vector<std::vector<float>> applyNMS(
    const std::vector<std::vector<float>>& boxes,
    int img_width,
    int img_height,
    float score_threshold,
    float nms_threshold
) {
    std::vector<cv::Rect> rects;
    std::vector<float> scores;

    for (const auto& box : boxes) {

        auto coords = yoloToCoordinates(
            box, img_width, img_height
        );

        int x1 = coords[0];
        int y1 = coords[1];
        int x2 = coords[2];
        int y2 = coords[3];

        rects.emplace_back(
            x1, y1,
            std::max(0, x2 - x1),
            std::max(0, y2 - y1)
        );

        scores.push_back(box[4]);
    }

    std::vector<int> indices;

    cv::dnn::NMSBoxes(
        rects,
        scores,
        score_threshold,
        nms_threshold,
        indices
    );

    std::vector<std::vector<float>> result;

    for (int idx : indices) {
        result.push_back(boxes[idx]);
    }

    return result;
}