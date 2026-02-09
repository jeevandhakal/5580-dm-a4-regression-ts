**1. What is supervised learning in general and prediction in
particular?**

![](media/image1.png){width="2.95in"
height="2.0665780839895014in"}Supervised learning is a core form of data
analysis used to extract models that describe important data classes or
forecast future trends. This method relies on a labeled dataset where
the algorithm is taught to recognize patterns based on known pairs of
inputs and outputs. Within this broad field, there is a critical
distinction between classification and prediction:

-   **Classification:** This process is used to predict categorical
    class labels. For instance, a bank loan officer might analyze
    historical data to determine if a new customer is \"risky\" or
    \"safe\". The goal here is to place the data point into a specific
    group.

-   **Prediction (Regression):** Prediction specifically refers to
    forecasting continuous-valued functions or numeric values. An
    example provided in the materials is a marketing manager who needs
    to predict the exact amount of money a given customer will spend
    during an upcoming sale.

-   **Business Context:** A marketing manager might use prediction to
    calculate the exact numeric value a customer will spend during a
    sale.

-   **Goal:** The ultimate objective of prediction is to use historical
    experience to make a statement about how a numeric variable will
    behave in the future.

**2. Examples of regression used in business applications such as
product demand.**

Regression analysis is widely used in the business world to identify the
relationship between two or more continuous variables, enabling
data-driven forecasting. The most common application is determining how
a change in one factor, like price or marketing, will influence an
outcome like sales or product demand.

-   **Italian Clothing Company Example:** In the provided case study, a
    clothing brand used linear regression to analyze the relationship
    between advertising spend and yearly sales . By designating sales as
    the response variable and advertising as the predictor, they
    estimated a precise relationship: Sales = 168 + 23 \* (Advertising)
    .

-   **Forecasting Demand:** Businesses use these models to anticipate
    product demand, which allows them to optimize inventory and supply
    chains. By analyzing historical demand levels against variables like
    pricing or economic indicators, companies can avoid both
    overstocking and missed sales opportunities.

-   **Strategic Planning:** Regression models can also predict customer
    lifetime value or the likely impact of a price hike on total
    revenue, helping managers make informed strategic decisions rather
    than relying on intuition.

**3. What are some of the popular prediction techniques?**

The field of machine learning offers various techniques for prediction,
ranging from simple statistical models to complex multi-layer
architecture. Each method has different ways of handling data and
minimizing errors:

-   **Linear Regression:** Useful for finding a direct relationship
    between a predictor (independent variable) and a response (dependent
    variable) . The model chooses values that minimize the error in the
    equation Y(pred) = b0 + b1\*x.

-   **Neural Networks:** These are multi-layer networks that simulate
    learning through hidden layers. They use **Forward Propagation** to
    calculate activations and **Backward Propagation** to tweak weights
    and biases based on errors.

-   **Support Vector Machine (SVM):** A linear model that creates a
    hyperplane to separate data for both classification and regression
    tasks.

-   **Regression Trees and Random Forests:** These techniques use a
    tree-like structure for decision-making. A Random Forest generalizes
    this by combining multiple trees, and performance can be refined by
    adjusting parameters like nodesize (e.g., changing it from 10,000 to
    1,000 to improve MAPE).

**4. What is time-series analysis?**

Time-series analysis is a mathematical procedure used to study data
points taken at specified times, usually at equal intervals. Unlike
standard regression, which looks at the relationship between different
variables, time-series analysis focuses on predicting future values
based solely on previously observed values of the same variable. A core
part of this analysis is **Time Series Decomposition**, which splits the
original data into three components:

-   **Seasonal Component:** This captures patterns that repeat within a
    fixed period . For example, a website might consistently see a spike
    in traffic during weekends.

-   **Trend Component:** This reflects the underlying long-term
    direction of the metrics . An example is a website that is gradually
    increasing in popularity over several years.

-   **Random/Noise:** These are the residuals left in the data after the
    trend and seasonal effects are removed . This analysis is essential
    for understanding whether a recent change is a temporary fluctuation
    or part of a larger trend.

![](media/image1.png){width="3.216666666666667in"
height="2.2916666666666665in"}

**5. What are some of the popular time-series prediction techniques?**

There are several specialized techniques for forecasting time-series
data, specifically designed to handle the unique challenges of trend and
seasonality:

-   **Holt-Winters:** This extension of exponential smoothing is
    designed to handle time series that contain both systematic trends
    and seasonal variations. It uses weighted sums of past observations,
    where recent observations are typically given more importance than
    those from the distant past.

-   **ARIMA(p, d, q):** Short for Auto Regressive Integrated Moving
    Average, this model is classified by three order parameters . p is
    the number of autoregressive terms, d is the number of non-seasonal
    differences needed, and q is the number of lagged forecast errors.

-   **Applied Examples:**

    -   **Beer Dataset:** The materials demonstrate performing
        Holt-Winters and ARIMA (4,0,0) on Australian beer consumption
        data to forecast the next 12 periods .

    -   **Electricity Demand:** An autoregression model can be built
        using variables such as the demand from the same hour last week,
        the same hour yesterday, and the previous four consecutive hours
        to predict the next hour's consumption .

**6. How do you measure the quality of prediction?**

To determine which model is the most effective, several metrics are used
to evaluate the error between predicted and actual values. The primary
metric highlighted in the presentations is the **Mean Absolute
Percentage Error (MAPE)**:

-   **Definition:** MAPE is the average of the absolute percentage
    errors of the forecasts. The error is calculated as the actual value
    minus the forecasted value.

-   **Why use absolute values?** Absolute values are used so that
    positive and negative errors do not cancel each other out, providing
    a true reflection of the model\'s accuracy.

-   **Benefits:** This measure is highly preferred in business contexts
    because it is easy to understand and provides the error in terms of
    a simple percentage.

-   ![](media/image1.png){width="3.25in"
    height="2.4763888888888888in"}**Model Comparison Example:** In the
    synthetic dataset experiments, different models showed varying
    degrees of accuracy. For example, a three-layer Neural Network
    achieved a significantly better test error (MAPE of 23.92%) compared
    to a Support Vector Machine, which had a MAPE of 69.42% .
    Ultimately, the smaller the MAPE, the better the forecast.

7.     Use the presentation to write a report on the beer dataset.

The beer dataset tracks monthly beer consumption in Australia from
January 1956 to August 1995. Based on the time-series analysis provided
in the lecture materials, this dataset serves as a perfect example of
how complex data can be broken down into predictable patterns for
business planning.

-   **Time Series Decomposition:** The primary step in analyzing this
    data is decomposition. This process separates the raw consumption
    figures into three distinct layers: the **Trend** (a steady
    long-term increase in consumption over decades), the **Seasonal**
    component (repeating annual cycles where consumption peaks during
    specific months), and **Random Noise** (unpredictable residuals).

-   **Modeling Techniques:**

    -   **Holt-Winters:** This method was used to handle the systematic
        trend and seasonal variations. The presentation shows that this
        model achieved a **MAPE of 5.44%**, making it a very reliable
        tool for forecasting.

    -   **ARIMA (4, 0, 0):** This autoregressive model was also applied
        to predict the next 12 periods. It uses four lags (past data
        points) to estimate future values.

-   **Key Learnings:** The analysis proves that beer consumption is not
    random but follows a highly seasonal cycle. For a brewery, this
    means they can accurately plan production and inventory levels 12
    months in advance with a high degree of confidence.
