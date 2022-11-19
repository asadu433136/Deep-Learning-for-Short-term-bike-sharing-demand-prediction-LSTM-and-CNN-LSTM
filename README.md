
# Project Title

Deep-Learning-for-Short-term-bike-sharing-demand-prediction-LSTM-and-CNN-LSTM.

## Description

These studies have shown how weather conditions can affect how many bikes are demanded at bike-sharing stations at times of the day and weeks. It will be easier for operating agencies to rebalance bike-sharing systems in a timely and efficient manner if we can predict the short-term demand for bike-sharing. We use Helsinki city historical trip data from 2016 to 2020, along with extra inputs such as weather conditions, demand-related engineered features, and temporal variables related to demand. As opposed to statistical methods, deep learning methods can automatically learn the relationship between inputs and outputs, requiring fewer assumptions and achieving higher accuracy. Since their spatiotemporal functional observations are similar, the CNN and LSTM need to be segmented into time frames like hourly, daily, weekly, and monthly, in order to model the Helsinki bike-sharing station data effectively. We provide quantitative results show that accurate short-term demand (CNN-LSTM_336) was noticeable between the Models.

## Getting Started

### Dataframe Implementation


### Dataset


* Bike-sharing Data

The bike-sharing dataset was provided by the local traffic agency Helsinki Region Transport (HRT), the operator of the bike-sharing system in Helsinki. City Bike Finland is a local company maintaining and rebalancing the shared bikes in the city. The dataset contains 12.138.008 entries with 14 attributes such as departure time, return time, departure id, departure name, distance, duration, avg speed, departure latitude, departure longitude, return latitude, return longitude, air temperature of Helsinki bike-sharing system, and so on.


* Weather Data

The data includes a single row of header information indicating the information in the column. The total number of records in the raw data was 39443 rows and 18 columns. (Visual crossing weather, 2020) The dataset includes features such as ‘Location’ and ‘Address’ which are diminished because the weather information covers the area of Helsinki. The features ‘Snow Depth’, ‘Wind dir’, and ‘Heat Index’ kept more than 50% of missing data or null data and are removed from the dataset, as the existing data is inaugurated to calculate any valid approximations numbers to replace the missing values. (Jessica Quach, 2008) ‘Wind Gust’, containing 30% missing values, is erased from the dataset while cleaning the dataset. The feature ‘conditions’, or weather types, also contain any of the values- clear, partially cloudy, rain, overcast, or ‘rain, partially cloudy’. The values of the feature are categorical and cannot be measured in a numerical way. Additionally, the values of this feature can be accounted for by the features ‘icon’ and ‘Soler radiation’ (Jessica Quach, 2008). In summary, ten features are accumulated for consideration as weather components for further preprocessing. Those are datetime, max-min and average temperature, dew, humidity, precipitation, windspeed, cloud cover, visibility, and solar energy.


* Dataset Pre-processing

Therefore, the first step was selecting & filtering some of the common data problems are missing data, duplicate rows, and column types. Both data sets were checked for these problems, with a slight variation between the two data sets given in their respective subsection. From the original bike-sharing dataset, we calculate the hourly number of pickups at each station. The timestamp is used as an index for bike-sharing and weather data, which helps to concatenate the dataset for prediction models. Hourly aggregation comes in a uniform format after processing for both datasets.

Framework

![image](https://user-images.githubusercontent.com/81937480/202842872-1b4a83c3-c84e-4995-9329-393097722fe3.png)

* Data Structure Design

Hourly aggregation demand of each station was processed into a demand data matrix as presented in the matrix (1). In which, 𝐷𝑡𝑐 denotes the number of pickup bikes at station c and tth time interval. We have 347 stations and 39.442 historical time intervals in our demand data. In weather matrix (2), 𝑊𝑡𝑓 denotes the weather condition f at tth time interval. From the analysis results of weather data obtained from Helsinki, we select seven crucial weather characteristics to use as weather features to train the prediction models.

* Input Data

To build the input data for the prediction models, we take the pickup demand at each station and stack it with shifted data for historical features. The number of historical demand features was decided by the number of "lags". Each lag represents 1 hour. We tested the performance of models with lags value: 12, 24 (1 day), 48 (2 days), 96 (3 days), 120 (5 days), 168 (1 week), 336 (2 weeks), 504 (3 weeks), 672 (4 weeks). For the weather features, we take the selected features at t-1, t, and forecasting weather conditions for the next 3 hours: t+1, t+2, and t+3.

![Capture](https://user-images.githubusercontent.com/81937480/202842973-147bf544-fbc9-4aa6-a148-e9571fba273d.JPG)

* A graphical illustration of feature builders for training and testing data

![image](https://user-images.githubusercontent.com/81937480/202842994-edecfc72-ee47-419b-ac80-565fc2bb1ec2.png)

* Output Data

Howbeit, we need vehicles to relocate bikes from low demand points to high demand points. It is, therefore, necessary to predict the demand at each station ahead. Hence, we will focus on determining which model predicts the most accurately and stably for short-term demand at the station in the time window of two hours. We decided to choose 2 hours as we believe this is the time it takes for the system to be able to arrange vehicles between stations. The target result is the total pickups of the next 2 hours at the respective station. The output array can be presented as:

![Capture 3](https://user-images.githubusercontent.com/81937480/202843183-74b052cb-abb0-4c57-82b2-1bb27aadae4d.JPG)






Heatmap of Correlation Matrix

![download (12)](https://user-images.githubusercontent.com/81937480/202843244-c0c1ef21-c1b6-4bf2-ba44-829f6dec830d.png)



### Methodology


### Proposed Methodology


* ARIMA & ARMA

* Long-Short Term memory (LSTM)

* Convolutional neural network(CNN)

* CNN -LSTM (Hybrid modelling)



### Experimental results and discussion

* Descriptive Analysis

The descriptive analysis of demand and weather data is consecutively drawn by several figures while visualizing the dataset.

* Demand Data
HSL Bike corporation counted bicycles from May 2016 to November 2020. this organization gathered 80% of the bicycle demand counts from May to October. The monthly (2017) bicycle count data figure is presented in Figure 9. More bicycle counts are observed during peak months, followed by summertime. To clarify, the average counts in 2017, May, June, July, August, September, and October are 10.12, 10.58, 9.59, 9.58, 7.59, and 5.06. Bicycle counts, as expected, reach their peak in July and their trough in January. As shown in the Figure, the yearly demand for bike sharing in Helsinki city is much more than the previous year. In 2016, the average pickup demand per year was max. 10.54, which is gradually increasing in 2017 is 43.17, in 2018 is 67.80, in 2019 is 86.99 and finally in 2020 is 73.74, slightly decreasing due to the pandemic Covid attacked by worldwide.

![image](https://user-images.githubusercontent.com/81937480/202844074-6ce8f3c2-3de6-489f-a39a-12bb0be9c1a5.png)

![image](https://user-images.githubusercontent.com/81937480/202844090-1afcf379-69bf-45a2-b21d-40100284c979.png)

![image](https://user-images.githubusercontent.com/81937480/202844101-9bcc8519-a79e-4ede-b5e1-0f5c35981b92.png)

*Weather data

From the statistical point of view, weather data, in contrast to average weather pickup demand, have a positive correlation. The average temperature records a slight decrease in 2017 (around -18.5°C), contrarily to the rainfall and relative humidity that display in noticeable increase, the intra-seasonal variation goes in concordance with the Visual crossing report. Where consecutively 2016,2018,2019and 2020 show a higher temperature than the year 2017. Such lousy weather is decreasing the demand for bike picking in Helsinki, are mostly had some bad weather. The figure shows the maximum and minimum temperature with median, Quartile 1, and Quartile 3.

![image](https://user-images.githubusercontent.com/81937480/202844150-68d42562-9429-49c4-bbdc-72991358059a.png)

* Aggregation of Datasets

The pickup duration distribution is upbringing in the middle, with the mode being around by month. In contrast, the average pickup count is affected by temperature for several years. The highest density can be observed between 4 to 9 months. Closer inspection of the pick duration trend by months, Figure 10 shows that there has been a steady decline from October to march each year, a gradual increase reaching its peak in April, then it drops again from October each year. The trip duration variability is most likely attributed to weather conditions (Temp, windspeed), as bike users tend to cycle more in warm conditions. Figure 10 confirms it; the same pattern can be seen, the gradual decline from January to February, then the continual growth reaching a peak in September.

![image](https://user-images.githubusercontent.com/81937480/202844188-5dc506d8-9131-4628-b76e-8ea0d0784a0d.png)

*  Stationary

This paper requires the analysis of both datasets from a statistical point of view. With this view, it was checked out that both datasets are stationary or non-stationary to implement the ARIMA model. It is checked by Plotting the ACF and PACF for both data sets. Also, use the Augmented Dickey-Fuller (ADF) test to check if both datasets are stationary or not. According to the ADF test, the time series is stationary. We can reject the hypothesis if the p-value is less than the significance level (0.05), thus indicating that the time series is stationary the result comes out with all 347-station data being stationary and the weather data being stationary. As in our case, it is unable to build ARIMA model for this research of predicting bike sharing demand. However, there is another forecasting model, ARMA, which is better fit because the datasets are stationary.
𝐴𝑅𝐼𝑀𝐴 (𝑝,𝑑,𝑞)=𝐴𝑅𝑀𝐴 (𝑝,𝑞)+ preliminary differencing procedure (𝑑)
An ARIMA (p, d, q) model is essentially an ARMA (p, q) model with roots in the d units. From its formula, it is easy to see:

![image](https://user-images.githubusercontent.com/81937480/202844312-8f2d4120-4258-48d1-83c0-03cc1b6aa679.png)

For the ARMA method, ACF and PACF clearly showed that the curves differ at each station in Figure 11. Eventually, for 347 stations, the method can be generalized for any models but at the cost of increased computational time. So, we put it in cue and turn our thought from the traditional statistical model to the deep learning model. However, the model (CNN&LSTM) order estimation accuracy is much better than it is for most of the aforementioned methods (ARMA & ARIMA) (Lei Ji, 2019). So, we ultimately chose the model CNN-LSTM and LSTM for further process.


*  Time Lags with LSTM and CNN-LSTM in Prediction Accuracy

Metric selection
We use mean absolute error (MAE) to measure the differences between actual and prediction values. Root mean squared error (RMSE) represents the root of the average squared difference between the original and predicted values. We also use it to measure the differences between actual and prediction values but amplify the effect of significant differences. However, our model not only helps to predict the demand of a particular station but of many stations and the demand of different stations are completely distinct. The median of average pickups at each station per hour of 347 stations is 0.772. Therefore, even small MAE and RMSE are not enough to conclude the performance of models. Meanwhile, R-squared represents the proportion of the variance for dependent variables. It could help explain the strength of the relationship between actual and prediction values. For mentioned reasons, we use R-squared as the primary metric to measure the accuracy of the models.

LSTM vs. CNN-LSTM
Due to the limited computing resources, we could only run our model for 200 stations. Therefore, the results from this part will apply to only 200 stations. We ran 20 models on Colab Pro+ of Google with 51GB of RAM. Each model needs 8 to 14 hours to complete the training and evaluate prediction results. Table 1 reports our testing results on average R-squared, average MAE, and average RMSE of 200 training stations. Comparing the corresponding results of the same time lag between LSTM and CNN-LSTM models, we see that 8 out of 10 models of CNN-LSTM give better average R-squared results. Although, for the time of 504 hours (3 weeks) and 672 hours (4 weeks), CNN-LSTM models perform lower average r-squared, these two models still give better RSME (smaller) values for CNN-LSTM. On this basis, we conclude that the hybrid CNN-LSTM model outperforms the conventional LSTM model in predicting short-term demand in the bike-sharing system for our setup. An explanation for this result could be that the CNN layer provides more robustness to capture parameters of training features for the conventional model.

![Capture 5](https://user-images.githubusercontent.com/81937480/202844394-6e8de3de-ba06-431d-a6ca-bc09169a055d.JPG)

Time lags
Figure 12 details the distribution of the R-squared value at each station in each prediction model. Based on table 1 and figure 12, we can observe that both the mean and median of all stations increase for both LSTM and CNN-LSTM as we increase the value of the lag time interval from 6 hours to 336 hours. This observation implies that the models will predict more accurately as we increase the time interval considered in the model's features. However, when the optimal value is reached, the model's performance will start to decrease. This can be explained that when the model reaches the optimal lag, increasing the lag value will increase the confounding information to the model. Based on the results obtained from the experiment, the optimal value for the historical demand period is 336 hours or two weeks.

![image](https://user-images.githubusercontent.com/81937480/202844404-eab25b03-9249-4e47-829f-6a55f986d5c4.png)

Average Pickups Demand in Prediction Accuracy

For the purpose of clustering stations, we use box-and-whisker to identify the distribution of the average demand of 200 stations. The upper whisker, upper hinge, median, lower hinge, and lower whisker are 3.435, 1.632, 0.875, 0.390, and 0.11, respectively. We use the upper and lower hinge values to divide stations into high demand, medium, and low demand groups. Hence 50% of stations will be in medium demand and 25% high and 25% on low demand.
We calculate the average of their mean demand for each demand group and compare it with the RMSE received from training our models for each hour. Figure 13 visualizes the results for CNN-LSTM 336 model. For high-demand stations, the average RMSE per hour is smaller than the average of mean demand by 0.685. These two values are roughly the same for the medium-demand group. However, in the low demand, the average of mean demand is only 54% of the value of the average RMSE. This finding reveals that the model's accuracy decreases according to the demand at each station. Thence, the model predicts more accurately for high-demand stations. We did the same calculations with all the remaining models and received the same result.

![image](https://user-images.githubusercontent.com/81937480/202844425-8d33e850-6eee-4b67-ad2e-c740aa3abcbd.png)


For deeper investigation, we run the logarithmic trendline model between r-squared and mean demand of stations. Figure 14 represents the correlation of the primary metric for model accuracy and the mean demand of each station. Appendix 6 reports the numerical result for the logarithmic equation for tested models. All the models get a significant logarithmic correlation with p-values less than 0.0001. The results of the experiment found clearly support for the previous discussion that the model predicts more accurately for high-density stations. It leads to good results in meeting customer expectations in crowded areas. This prediction ability is crucial in increasing satisfaction and maintaining the sustainability of bike-sharing system's growth.

![image](https://user-images.githubusercontent.com/81937480/202844449-cef4f379-f303-4cac-8f1a-c1c28b4d03b5.png)

Weather Forecasting in Prediction Accuracy

In the previous section, we concluded that the period interval considering for pickup demand is important, and 336 hours is the optimal lags for prediction models. In that experimental design, we used 1-hour historical weather data, the current weather and forecasting weather in the next 3 hours. This setup is because we believe customers need time to prepare to go out by bike, and the delay is one hour. Besides, users also check the upcoming short-term weather forecast to decide if it is suitable for cycling. However, another question arises about how many hours the user will look at the weather forecast to make a decision. To answer this question, we redesigned our experiment with historical demand in 336 hours and changed the amount of weather information in the coming hours, starting at 2 hours, 3 hours, 4 hours, and 5 hours respectively. Table 2 shows the results of our new experiments.
As initially predicted, the increase or decrease in the amount of weather information affects the accuracy of the model's prediction results. We can observe the pattern more accurately with increasing average r-squared, and MAE, RMSE decreasing as the amount of weather information is increased from 2 hours to 4 hours. Then the accuracy decreases as we increase to 5 hours for both LSTM and CNN-LSTM models. This finding can be explained as users only look at weather forecasts for their trips for the next 4 hours. It also corresponds to the analysis of the user's trip duration.

![Capture 6](https://user-images.githubusercontent.com/81937480/202844473-d2a4fd3c-ebcf-4a27-8cb0-01f744d0a4d0.JPG)


## Conclusion

Discussion for Future Research
Despite the limitations of our computing resources, we optimized our code and model to pre-process the huge dataset (12 million rows) and train the prediction models. Our research experiment results provide solid evidence to select the proper timeframe for the deep learning model. We approach the topic using traditional statistical analysis techniques to analyze Spatiotemporal data properties to show the traditional method's limitations. Furthermore, we implement modern deep learning models to evaluate relationships of natural factors affecting the model's accuracy. Our model is able to reach 90% accuracy for some stations and achieves an average error for the whole bike-sharing system of 2 pickups for 2 hours at each station or one pickup per hour. This result is promising. Our research has shown that the CNN-LSTM model with a lag interval of 336 hours and 4 hours of weather forecast conditions gives the highest accuracy at RMSE of 2.182 for 2 hours and average R-Squared for 200 stations at 0.435. This provides a good starting point for discussion to develop a good prediction model.
From the results of this study, future research should consider the return bikes at each station to predict the net demand at each station. In addition, the model can be developed into professional analytical applications to support existing prediction systems.

## Final Result

Boxplot distribution of MAE results for 200 stations

![image](https://user-images.githubusercontent.com/81937480/202844506-506a7f17-fcda-4668-aff7-a669a86978ed.png)


Boxplot distribution of RMSE results for 200 stations

![image](https://user-images.githubusercontent.com/81937480/202844548-e0f79b59-1f38-418b-99aa-f1c73a211b9b.png)

Sample prediction result at Erottajan Aukio station

![image](https://user-images.githubusercontent.com/81937480/202844581-10bd4dda-bf77-400a-bdcb-67fe19e9fb03.png)

Average R-Squared, MAE, RMSE of testing model for high, medium, and low demand stations

![image](https://user-images.githubusercontent.com/81937480/202844633-92e9cacd-daed-4522-920f-b08d6ea8c1a7.png)

## License

This project is licensed under the [OVGU-PROF] License - see the LICENSE.md file for details
