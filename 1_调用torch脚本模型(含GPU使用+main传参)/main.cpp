#include<iostream>
#include<torch/script.h>
#include<memory>

int main(int argc,const char* argv[])
{
	//�鿴�����Ƿ�Ϸ�
	std::cout << argc << std::endl; //argv�����±���1��ʼ���±�0��main������
	if (argc != 2)
	{
		std::cerr << "user:name.cpp model_path\n";
		system("pause");
		return -1;
	}

	//����ģ��
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

	//����device
	torch::DeviceType device_type=torch::kCUDA;
	torch::Device device(device_type, 0);

	//ģ���Լ���������ת�Ƶ�device��
	module.to(device);
	std::vector<torch::jit::IValue> input;
	input.push_back(torch::ones({ 1,3,224,224 }).to(device));

	//����ģ�ͣ���ӡ���ֽ��
	at::Tensor output = module.forward(input).toTensor(); //��������ת��Ϊtensor
	std::cout << output.slice(/*dim=*/1, /*start=*/0, /*end=*/5) << std::endl;
	
	system("pause");
}