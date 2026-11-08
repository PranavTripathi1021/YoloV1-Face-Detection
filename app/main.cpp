#include<iostream>
#include<opencv2/opencv.hpp>
#include<opencv2/dnn.hpp>
#include<torch/script.h>
#include<torch/cuda.h>
#include<vector>
#include<algorithm>
#include "functions.hpp"

static int S = 7;
static int B = 2;
static int C = 1;
float CONF_THRESHOLD = 0.15;
float NMS_THRESHOLD = 0.30;


int main(){
    cv::VideoCapture video(0);
    if (!video.isOpened()){
        std::cout<<"Video not found!";
        exit(0);
    }
    else{
        double frameRate = video.get(cv::CAP_PROP_FPS);
        std::cout<<"FPS is "<<frameRate<<'\n';
    }

    torch::jit::script::Module model;
    torch::Device device = (torch::cuda::is_available())? torch::kCUDA : torch::kCPU;
    // torch::Device device = torch::kCPU;
    try{
        model = torch::jit::load("../../../model/model.pt");
        model.to(device);
        model.eval();
        
    }
    catch(const c10::Error& e){
        std::cout<<"Coudn't load model: "<<e.what()<<'\n';
        exit(0);
    }

    std::cout<<"Model LOaded...\n";


    while(video.isOpened()){
        cv::Mat frame;
        bool succ = video.read(frame);
        if (!succ || frame.empty()) break;


        torch::Tensor img_tensor = imgToTensor(frame);
        img_tensor = img_tensor.to(device);
        torch::InferenceMode no_grad;
        std::vector<torch::jit::IValue> input;
        input.push_back(img_tensor);
        torch::Tensor predictions = model.forward(input).toTensor().to(torch::kCPU).detach().contiguous().to(torch::kFloat32);
        predictions = predictions.reshape({1, S, S, C + 5*B});
        predictions = predictions.squeeze(0);

        if (predictions.dim() != 3 ||
            predictions.size(0) != S ||
            predictions.size(1) != S ||
            predictions.size(2) != 11) {

            std::cerr << "Unexpected model output shape!\n";
            break;
        }

        std::vector<std::vector<float>> boxes = predToBoxes(predictions, CONF_THRESHOLD, S, C, B);
        boxes = applyNMS(boxes, frame.cols, frame.rows, CONF_THRESHOLD, NMS_THRESHOLD);

        for (const auto& box : boxes) {

            auto coords = yoloToCoordinates(
                box,
                frame.cols,
                frame.rows
            );

            cv::rectangle(
                frame,
                cv::Point(coords[0], coords[1]),
                cv::Point(coords[2], coords[3]),
                cv::Scalar(0, 255, 0),
                2
            );
        }
        cv::imshow("Face Detection", frame);

        int key = cv::waitKey(20);
        if(key == 'q'){
            std::cout<<"Stopping the video\n";
            break;
        }
    }
    video.release();
    cv::destroyAllWindows();
}
