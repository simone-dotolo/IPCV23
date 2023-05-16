<br />
<div align="center">
  
  <img src="https://github.com/simone-dotolo/PantasticSharpening/blob/main/media/pansharpening.gif"/>
  <h3 align="center">Pan-tastic Images and How to Sharpen Them</h3>

</div>

<!-- ABOUT THE PROJECT -->
## About The Project

This project was developed as part of the final exam of the Image Processing for Computer Vision course from the University of Naples Federico II. 

<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

_You should install Anaconda and Git before continuing._

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/simone-dotolo/PantasticSharpening.git
   ```
2. Create the virtual environment with the pantastic.yml
   ```sh
   conda env create -n pantastic -f pantastic.yml
   ```
   
### Usage

1. Activate the Conda Environment
   ```sh
   conda activate pantastic
   ```
2. Try it!
   ```sh
   python3 pansharpening.py -m APNN -s WV3 -i example/WV3_example.mat -o ./ -w weights/APNN_weights.pth
   ```
   
<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Simone Dotolo - sim.dotolo@gmail.com

LinkedIn: [https://www.linkedin.com/in/simone-dotolo/](https://www.linkedin.com/in/simone-dotolo/)

<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

This project was inspired by the following papers:
- [Pansharpening by Convolutional Neural Networks](https://www.mdpi.com/2072-4292/8/7/594)
- [Target-adaptive CNN-based pansharpening](https://arxiv.org/abs/1709.06054)
- [Pansharpening by convolutional neural networks in the full resolution framework](https://arxiv.org/abs/2111.08334)
- [Full-Resolution Quality Assessment for Pansharpening](https://www.mdpi.com/2072-4292/14/8/1808)
- [Fast Full-Resolution Target-Adaptive CNN-Based Pansharpening Framework](https://www.mdpi.com/2072-4292/15/2/319)
