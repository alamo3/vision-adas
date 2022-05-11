#Vision Radar

This project attempts to calculate lane lines and information about the lead car based on two sequential input
camera frames. Special thanks to [comma.ai](https://comma.ai) for making their [openpilot](https://github.com/commaai/openpilot)
model open-source for us all to use. Also special thanks to [@littlemountainman](https://github.com/littlemountainman) 
and [@nikebless](https://github.com/nikebless) for a lot of the project code [modeld](https://github.com/littlemountainman/modeld)
, [openpilot-pipeline](https://github.com/nikebless/openpilot-pipeline).

##Instructions to run

###Run the following commands:
Make sure to have python and pip3 installed beforehand.

Install the required packages:
- pip3 install onnxruntime
- pip3 install numpy
- pip3 install opencv-python

Run main.py:
- python3 main.py