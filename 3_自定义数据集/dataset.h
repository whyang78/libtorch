#ifndef __DATASET_H_
#define __DATASET_H_

#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<string>
#include<vector>

struct Options
{
	int image_size = 224;
	size_t batch_size = 32;
	double learning_rate = 0.01;
	size_t Epoch = 10;
	double best_accuracy = 0.0;

	std::string root_path = "./101_ObjectCategories/";
	std::string info_path = "./101_ObjectCategories/info.txt";

	torch::DeviceType devicetype = torch::kCPU;
}options;

//定义数据类型
using Data = std::vector<std::pair<std::string, long>>;

//读取info.txt
std::pair<Data, Data> readInfo()
{
	Data train, test;

	std::ifstream stream(options.info_path);
	assert(stream.is_open());

	std::string path, type;
	long label;

	while (true)
	{
		stream >> path >> label >> type;

		if (type == "train")
		{
			train.push_back(std::make_pair(path, label));
		}
		else if (type == "test")
		{
			test.push_back(std::make_pair(path, label));
		}
		else
		{
			assert(false);
		}

		if (stream.eof())
		{
			break;
		}
	}
	std::random_shuffle(train.begin(), train.end());
	std::random_shuffle(test.begin(), test.end());

	return std::make_pair(train, test);
}

//自定义数据集
class customDataset : public torch::data::Dataset<customDataset>
{
private:
	Data data;

public:
	//初始化
	customDataset(const Data& data) :data(data) {}

	//get example
	torch::data::Example<> get(size_t index) override
	{
		//std::string path = options.root_path + data[index].first;

		//cv::Mat image = cv::imread(path);
		//assert(!image.empty());
		//cv::resize(image, image, cv::Size(options.image_size, options.image_size), cv::INTER_CUBIC);
		//cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

		//
		//torch::Tensor image_tensor = torch::from_blob(image.data, { image.rows,image.cols,3 }, torch::kFloat);//0-255
		////torch::Tensor image_tensor= torch::from_blob(image.data, { image.rows,image.cols,3 }, torch::kFloat32).div(255);//0-1
		//image_tensor.permute({ 2,0,1 });
		//torch::Tensor label_tensor = torch::from_blob(&data[index].second, { 1 }, torch::kLong);

		//return { image_tensor,label_tensor };

		std::string path = options.root_path + data[index].first;
		auto mat = cv::imread(path);
		assert(!mat.empty());

		cv::resize(mat, mat, cv::Size(options.image_size, options.image_size));
		std::vector<cv::Mat> channels(3);
		cv::split(mat, channels);

		auto R = torch::from_blob(
			channels[2].ptr(),
			{ options.image_size, options.image_size },
			torch::kUInt8);
		auto G = torch::from_blob(
			channels[1].ptr(),
			{ options.image_size, options.image_size },
			torch::kUInt8);
		auto B = torch::from_blob(
			channels[0].ptr(),
			{ options.image_size, options.image_size },
			torch::kUInt8);

		auto tdata = torch::cat({ R, G, B })
			.view({ 3, options.image_size, options.image_size })
			.to(torch::kFloat);
		auto tlabel = torch::from_blob(&data[index].second, { 1 }, torch::kLong);
		return { tdata, tlabel };

	}

	//get size
	torch::optional<size_t> size() const override
	{
		return data.size();
	}
};


#endif // !__DATASET_H_

