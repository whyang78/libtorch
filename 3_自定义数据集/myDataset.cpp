#include<torch/torch.h>
#include<opencv2/opencv.hpp>
#include<iostream>
#include<memory>
#include<fstream>
#include"dataset.h"

//定义网络
struct NetImpl : torch::nn::SequentialImpl {
	NetImpl() {
		using namespace torch::nn;

		auto stride = torch::ExpandingArray<2>({ 2, 2 });
		torch::ExpandingArray<2> shape({ -1, 256 * 6 * 6 });
		push_back(Conv2d(Conv2dOptions(3, 64, 11).stride(4).padding(2)));
		push_back(Functional(torch::relu));
		push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
		push_back(Conv2d(Conv2dOptions(64, 192, 5).padding(2)));
		push_back(Functional(torch::relu));
		push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
		push_back(Conv2d(Conv2dOptions(192, 384, 3).padding(1)));
		push_back(Functional(torch::relu));
		push_back(Conv2d(Conv2dOptions(384, 256, 3).padding(1)));
		push_back(Functional(torch::relu));
		push_back(Conv2d(Conv2dOptions(256, 256, 3).padding(1)));
		push_back(Functional(torch::relu));
		push_back(Functional(torch::max_pool2d, 3, stride, 0, 1, false));
		push_back(Functional(torch::reshape, shape));
		push_back(Dropout());
		push_back(Linear(256 * 6 * 6, 4096));
		push_back(Functional(torch::relu));
		push_back(Dropout());
		push_back(Linear(4096, 4096));
		push_back(Functional(torch::relu));
		push_back(Linear(4096, 102));
		push_back(Functional(torch::log_softmax, 1, torch::nullopt));
	}
};
TORCH_MODULE(Net); //注意与类名称的区别

template<typename DataLoader>
void train(size_t epoch,Net& net,torch::optim::Optimizer& optimizer,DataLoader& dataloader,
	size_t data_size, torch::Device& device)
{
	size_t batchidx = 0;
	net->train();
	for (auto& batch : dataloader)
	{
		auto data = batch.data.to(device);
		auto target = batch.target.to(device).view({ -1 });
		auto output = net->forward(data);
		auto loss = torch::nll_loss(output, target);
		AT_ASSERT(!std::isnan(loss.item<float>()));

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();


		if (batchidx++ % 10 == 0)
		{
			std::printf("\r epoch:%ld,[%6ld/%6ld],batch:%ld,loss:%.4f", 
				epoch,batchidx*batch.data.size(0),data_size, batchidx, loss.item<float>());
		}
	}
}

template<typename DataLoader>
void test(Net& net, DataLoader& dataloader, size_t data_size, torch::Device& device)
{
	net->eval();
	torch::NoGradGuard no_grad;
	double test_loss = 0.0;
	int64_t correct = 0;
	for (auto& batch : dataloader)
	{
		auto data = batch.data.to(device);
		auto target = batch.target.squeeze().to(device);
		auto output = net->forward(data);
		auto loss = torch::nll_loss(output, target);
		
		auto pred = torch::argmax(output, 1);
		correct += torch::eq(pred, target).sum().item<int64_t>();
		test_loss += loss.item<float>();
	}

	double accuracy = static_cast<double>(correct) / data_size;
	std::printf("test loss:%.4f, accuracy:%.2f", test_loss, accuracy);
	if (accuracy > options.best_accuracy)
	{
		options.best_accuracy = accuracy;
		torch::save(net, "best_net.pt");
	}
}


int main()
{
	torch::manual_seed(78);

	if (torch::cuda::is_available())
	{
		options.devicetype = torch::kCUDA;
	}
	torch::Device device(options.devicetype, 0);

	//读取info.txt
	auto data = readInfo();

	//训练数据集
	auto trainData = customDataset(data.first).map(torch::data::transforms::Stack<>());
	auto train_size = trainData.size().value();
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(trainData), options.batch_size);

	//测试数据集
	auto testData = customDataset(data.second).map(torch::data::transforms::Stack<>());
	auto test_size = testData.size().value();
	auto testDataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(testData), options.batch_size);

	//加载网络
	Net net;
	net->to(device);
	torch::optim::SGD optimizer(net->parameters(), torch::optim::SGDOptions(options.learning_rate).momentum(0.5));
	
	//训练与测试
	for (size_t epoch = 1; epoch <= options.Epoch; epoch++)
	{
		train(epoch, net, optimizer, *trainDataLoader, train_size,device);
		test(net, *testDataLoader, test_size,device);
	}

	system("pause");
}