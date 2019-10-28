#include<iostream>
#include<torch/torch.h>
#include<string>
#include<memory>
#include<string>
#include"net.h"

const int64_t batch_size = 32;
const double lr = 0.01;
const int64_t Epoch = 10;
double best_accuracy = 0.0;



//ѵ������
template<typename DataLoader, typename Model>
void train(int64_t epoch, Model& model, DataLoader& dataloader,torch::optim::Optimizer& optimizer,
	size_t data_size, torch::Device& device)
{
	model->train();
	size_t batchidx = 0;
	for (auto& batch : dataloader) //��dataloader��ȡbatch
	{
		auto data = batch.data.to(device), target = batch.target.to(device);
		auto output = model->forward(data);
		auto loss = torch::nll_loss(output, target);
		AT_ASSERT(!std::isnan(loss.item<float>()));

		optimizer.zero_grad();
		loss.backward();
		optimizer.step();
		
		batchidx += 1;
		if (batchidx % 10 == 0)
		{
			std::printf("\r train epoch:%ld ,[%6ld/%6ld],loss:%.4f",
				epoch,
				batchidx*batch.data.size(0),
				data_size,
				loss.item<float>());
		}
	}
}

//���Ժ���
template<typename DataLoader,typename Model>
void test(Model& model, DataLoader& dataloader, size_t data_size,torch::Device& device)
{
	torch::NoGradGuard no_grad;
	model->eval();

	int64_t correct = 0;
	double test_loss = 0.0;
	for (auto& batch : dataloader)
	{
		auto data = batch.data.to(device), target = batch.target.to(device);
		auto output = model->forward(data);
		auto loss = torch::nll_loss(output, target);

		auto pred = torch::argmax(output, 1);
		correct += torch::eq(pred, target).sum().item<int64_t>();
		test_loss += loss.item<float>();
	}

	double accuracy = static_cast<double>(correct) / data_size;//ǿ������ת��
	std::printf("\ntest loss:%.4f, accuracy:%.4f\n", test_loss, accuracy);

	if (accuracy > best_accuracy)
	{
		best_accuracy = accuracy;
		torch::save(model, "best_model.pt");//����ģ��
	}
}

int main(int argc, const char *argv[])
{
	if (argc != 2)
	{
		std::cerr << "�޲���:dataset_path" << '\n';
		system("pause");
		return -1;
	}

	//����device
	torch::DeviceType device_type;
	if (torch::cuda::is_available())
	{
		device_type = torch::kCUDA;
	}
	else
	{
		device_type = torch::kCPU;
	}
	torch::Device device(device_type, 0);

	//�������������
	torch::manual_seed(78);

	//����ģ��
	auto model=std::make_shared<vggNet>();//��Ҫ����ģ�ͣ���ʼ��ģ��ʱ��һ��ҪΪָ���ʽ
	model->to(device);

	//�������ݼ�
	std::string dataset_path = argv[1];
	auto trainData = torch::data::datasets::MNIST(dataset_path, torch::data::datasets::MNIST::Mode::kTrain)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	size_t train_size = trainData.size().value();
	auto trainDataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(trainData), batch_size);

	auto testData = torch::data::datasets::MNIST(dataset_path, torch::data::datasets::MNIST::Mode::kTest)
		.map(torch::data::transforms::Normalize<>(0.1307, 0.3081))
		.map(torch::data::transforms::Stack<>());

	size_t test_size = testData.size().value();
	auto testDataLoader = torch::data::make_data_loader<torch::data::samplers::SequentialSampler>(
		std::move(testData), batch_size);

	//�����Ż���
	torch::optim::SGD optimizer(model->parameters(), torch::optim::SGDOptions(lr).momentum(0.5));

	//ѵ�������Թ���
	for (int64_t epoch = 1; epoch <= Epoch; epoch++)
	{
		train(epoch, model, *trainDataLoader, optimizer, train_size,device);
		test(model, *testDataLoader, test_size, device);
	}
	
	system("pause");

}

