# Project: Machine learning models to predict house prices based on Boston House Pricing dataset  
# Version: 1.0  
# Date: 1/31/2021  
# Description: Flask based API server to serve ML model  
# Author: Saranya Murugan  
  
  
## Folder tree:  
root  
	models  
	- final_model.pkl  
	- x_scaler.pkl  
	- y_scaler.pkl  
	.gitignore  
	main.py  
	README.md  
	requirements.txt  
  
  
## Instructions to run the inference model:  
1. Install virtual environment  
pip install virtualenv  
2. Create the virtual environment  
virtualenv venv  
3. Active the virtual environment  
.\venv\Scripts\activate
4. Clone this repository using  
git clone https://github.com/samuruga/Boston-House-Price-Prediction.git  
5. Navigate to the project root  
cd Boston-House-Price-Prediction  
6. Install project requirements  
pip install -r requirements.txt  
7. Run the api server  
python main.py  
8. Use a browser to browse to the swagger page  
http://localhost:5000/apidocs  
9. Select the '/predict' api request, click 'Try it out', upload the test csv dataset and click 'Execute'  
  