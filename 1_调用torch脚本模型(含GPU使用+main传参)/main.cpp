#include<iostream>
#include<torch/script.h>
#include<memory>

int main(int argc,const char* argv[])
{
	//查看参数是否合法
	std::cout << argc << std::endl; //argv参数下标自1开始，下标0是main函数名
	if (argc != 2)
	{
		std::cerr << "user:name.cpp model_path\n";
		system("pause");
		return -1;
	}

	//载入模型
	torch::jit::script::Module module;
	try
	{
		module = torch::jit::load(argv[1]);
	}
	catch (const c10::Error& e)
	{
		std::cerr << "model load failed!\n";
		system("pause");
		return -1;
	}
	std::cout << "model load successfully!\n";

	//设置device
	torch::DeviceType device_type=torch::kCUDA;
	torch::Device device(device_type, 0);

	//模型以及输入数据转移到device上
	module.to(device);
	std::vector<torch::jit::IValue> input;
	input.push_back(torch::ones({ 1,3,224,224 }).to(device));

	//运行模型，打印部分结果
	at::Tensor output = module.forward(input).toTensor(); //数据类型转换为tensor
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
	
	system("pause");
}