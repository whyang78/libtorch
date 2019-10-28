#ifndef __NET_H_
#define __NET_H_

#include<torch/torch.h>

//vGGÍøÂçÐÔÄÜÌ«²î£¬ÍøÂçÌ«¸´ÔÓ
struct vggNet: torch::nn::Module
{
	vggNet()
	{
		conv1_1 = register_module("conv1_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 64, 3).padding(1)));
		conv1_2 = register_module("conv1_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 64, 3).padding(1)));
		// Insert pool layer
		conv2_1 = register_module("conv2_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(64, 128, 3).padding(1)));
		conv2_2 = register_module("conv2_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 128, 3).padding(1)));
		// Insert pool layer
		conv3_1 = register_module("conv3_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(128, 256, 3).padding(1)));
		conv3_2 = register_module("conv3_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		conv3_3 = register_module("conv3_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 256, 3).padding(1)));
		// Insert pool layer
		conv4_1 = register_module("conv4_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(256, 512, 3).padding(1)));
		conv4_2 = register_module("conv4_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		conv4_3 = register_module("conv4_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		// Insert pool layer
		conv5_1 = register_module("conv5_1", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		conv5_2 = register_module("conv5_2", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));
		conv5_3 = register_module("conv5_3", torch::nn::Conv2d(torch::nn::Conv2dOptions(512, 512, 3).padding(1)));

		fc1 = register_module("fc1", torch::nn::Linear(512, 256));
		fc2 = register_module("fc2", torch::nn::Linear(256, 128));
		fc3 = register_module("fc3", torch::nn::Linear(128, 10));
		
	}

	torch::Tensor forward(torch::Tensor x)
	{
		x = torch::relu(conv1_1->forward(x));
		x = torch::relu(conv1_2->forward(x));
		x = torch::max_pool2d(x,2);

		x = torch::relu(conv2_1->forward(x));
		x = torch::relu(conv2_2->forward(x));
		x = torch::max_pool2d(x, 2);

		x = torch::relu(conv3_1->forward(x));
		x = torch::relu(conv3_2->forward(x));
		x = torch::relu(conv3_3->forward(x));
		x = torch::max_pool2d(x, 2);

		x = torch::relu(conv4_1->forward(x));
		x = torch::relu(conv4_2->forward(x));
		x = torch::relu(conv4_3->forward(x));
		x = torch::max_pool2d(x, 2);

		x = torch::relu(conv5_1->forward(x));
		x = torch::relu(conv5_2->forward(x));
		x = torch::relu(conv5_3->forward(x));
		
		x = x.view({ -1,512 });
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = fc3->forward(x);
		x = torch::log_softmax(x,1);

		return x;
	}

	torch::nn::Conv2d conv1_1{ nullptr };
	torch::nn::Conv2d conv1_2{ nullptr };
	torch::nn::Conv2d conv2_1{ nullptr };
	torch::nn::Conv2d conv2_2{ nullptr };
	torch::nn::Conv2d conv3_1{ nullptr };
	torch::nn::Conv2d conv3_2{ nullptr };
	torch::nn::Conv2d conv3_3{ nullptr };
	torch::nn::Conv2d conv4_1{ nullptr };
	torch::nn::Conv2d conv4_2{ nullptr };
	torch::nn::Conv2d conv4_3{ nullptr };
	torch::nn::Conv2d conv5_1{ nullptr };
	torch::nn::Conv2d conv5_2{ nullptr };
	torch::nn::Conv2d conv5_3{ nullptr };

	torch::nn::Linear fc1{ nullptr }, fc2{ nullptr }, fc3{ nullptr };
};

struct Net : torch::nn::Module {
	Net()
	 {
		conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 10, /*kernel_size=*/5)));
		conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(10, 20, /*kernel_size=*/5)));
		fc1 = register_module("fc1", torch::nn::Linear(320, 50));
		fc2 = register_module("fc2", torch::nn::Linear(50, 10));

		register_module("conv2_drop", conv2_drop);
	}

	torch::Tensor forward(torch::Tensor x) {
		x = torch::max_pool2d(conv1->forward(x), 2);
		x = torch::relu(x);

		x = conv2_drop->forward(conv2->forward(x));
		x = torch::max_pool2d(x, 2);
		x = torch::relu(x);

		x = x.view({ -1, 320 });
		x = torch::relu(fc1->forward(x));
		x = torch::dropout(x, /*p=*/0.5, /*training=*/is_training());
		x = fc2->forward(x);
		return torch::log_softmax(x, /*dim=*/1);
	}

	torch::nn::Conv2d conv1{ nullptr };
	torch::nn::Conv2d conv2{ nullptr };
	torch::nn::FeatureDropout conv2_drop;
	torch::nn::Linear fc1{ nullptr };
	torch::nn::Linear fc2{ nullptr };
};

struct LinearNet : torch::nn::Module {
	LinearNet()
	{
		fc1 = register_module("fc1", torch::nn::Linear(28 * 28, 512));
		fc2 = register_module("fc2", torch::nn::Linear(512, 128));
		fc3 = register_module("fc3", torch::nn::Linear(128, 10));
	}

	torch::Tensor forward(torch::Tensor x) {
		x = x.view({ -1, 28 * 28 });
		x = torch::relu(fc1->forward(x));
		x = torch::relu(fc2->forward(x));
		x = fc3->forward(x);
		return torch::log_softmax(x, /*dim=*/1);
	}

	torch::nn::Linear fc1{ nullptr };
	torch::nn::Linear fc2{ nullptr };
	torch::nn::Linear fc3{ nullptr };
};

#endif // !

