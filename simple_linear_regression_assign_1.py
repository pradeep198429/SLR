#!/usr/bin/env python
# coding: utf-8

#  Calories_consumed-> predict weight gained using calories consumed.
#     Do the necessary transformations for input variables for getting better R^2 value for the model prepared.
# 

# In[2]:


import pandas as pd ## importing necessary libraries pandas


# In[3]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


#  reading a csv file using pandas library

# In[13]:


calories=pd.read_csv("calories_consumed_new.csv")


# In[14]:


calories.columns


# In[15]:


plt.hist(calories.Weight_gained_in_grams)


# by above histogram we can conclude that our data is not normalized

# In[16]:


plt.boxplot(calories.Weight_gained_in_grams,0,"rs",0)


# it is also shown that our data is not normalized

# In[17]:


plt.hist(calories.Calories_consumed)


# it shows that calories consumed is not normal

# In[18]:


plt.boxplot(calories.Calories_consumed)


# box blot also showing that calories consumed is not normalized

# In[19]:


plt.plot(calories.Calories_consumed,calories.Weight_gained_in_grams,"bo")
plt.xlabel("Calories_consumed")
plt.ylabel("Weight_gained_in_grams")


# # For preparing linear regression model we need to import the statsmodels.formula.api

# In[20]:


import statsmodels.formula.api as smf


# In[21]:


model=smf.ols("Weight_gained_in_grams~Calories_consumed",data=calories).fit()


#  For getting coefficients of the varibles used in equation

# In[22]:


model.params


# P-values for the variables and R-squared value for prepared model

# In[23]:


model.summary()


# In[24]:


pred=model.predict(calories)


# In[25]:


pred


# In[26]:


#pred = model.predict(calories.iloc[:,1]) # Predicted values of Weight_gained_in_grams using the model here we are predecting 
#weight_gained_grams using calories_consumed which is first column thats why i passed [:,1]


# In[27]:


#pred=model.predict(pd.DataFrame(calories['Calories_consumed'])) #predicted values of Weight_gained_in_grams using the model
#here we r predicting Weight_ganed_in_grams using Calories_consumed 


# In[28]:


resid_error=pred-calories.Weight_gained_in_grams  #resid_error=prediction-actual y


# In[29]:


resid_error


# In[30]:


rmse_model=np.sqrt(np.mean(resid_error**2))


# In[31]:


rmse_model


# In[32]:


calories.Weight_gained_in_grams.corr(calories.Calories_consumed) # # correlation value between X and Y


# In[33]:


np.corrcoef(calories.Weight_gained_in_grams,calories.Calories_consumed)


# In[34]:


model.conf_int(0.05) # 95% confidence interval


#  Visualization of regresion line over the scatter plot of Waist and AT
#  For visualization we need to import matplotlib.pyplot

# In[35]:


import matplotlib.pylab as plt


# In[36]:


plt.scatter(x=calories['Calories_consumed'],y=calories['Weight_gained_in_grams'],color='red');plt.plot(calories['Calories_consumed'],pred,color='black');plt.xlabel('Calories_consumed');plt.ylabel('Weight_gained_in_grams')


# In[37]:


pred.corr(calories.Weight_gained_in_grams)


#  Transforming variables for accuracy

# In[38]:


model2 = smf.ols('Weight_gained_in_grams~np.log(Calories_consumed)',data=calories).fit()


# In[39]:


model2.params


# In[40]:


model2.summary()


# In[41]:


pred2 = model2.predict(calories)


# In[42]:


#pred2=model2.predict(pd.DataFrame(calories['Calories_consumed']))


# In[43]:


#pred=model2.predict(calories.iloc[:,1])


# In[44]:


pred2


# In[45]:


resid_error2=pred2-calories['Weight_gained_in_grams']


# In[46]:


rmse_model2=np.sqrt(np.mean(resid_error2**2))


# In[47]:


rmse_model2


# In[48]:


pred2.corr(calories.Weight_gained_in_grams)


# its less than model1

# In[49]:


plt.scatter(x=calories['Calories_consumed'],y=calories['Weight_gained_in_grams'],color='red');plt.plot(calories['Calories_consumed'],pred2,color='black');plt.xlabel('Calories_consumed');plt.ylabel('Weight_gained_in_grams')


# now we will develop Exponential transformation

# In[50]:


model3 = smf.ols('np.log(Weight_gained_in_grams)~Calories_consumed',data=calories).fit()


# In[51]:


model3.params


# In[52]:


model3.summary() #its r-sq value is less than model


# In[53]:


print(model3.conf_int(0.01)) # 99% confidence level


# In[54]:


pred_log = model3.predict(pd.DataFrame(calories['Calories_consumed']))


# In[55]:


#pred_log= model3.predict(calories.iloc[:,1])


# In[56]:


#pred_log=model3.predict(calories)


# In[57]:


pred_log


# In[58]:


pred3=np.exp(pred_log)  # as we have used log(Weight_gained_in_grams) in preparing model so we need to convert it back


# In[59]:


pred3


# In[60]:


pred3.corr(calories.Weight_gained_in_grams)


# In[61]:


plt.scatter(x=calories['Calories_consumed'],y=calories['Weight_gained_in_grams'],color='red');plt.plot(calories['Calories_consumed'],pred3,color='black');plt.xlabel('Calories_consumed');plt.ylabel('Weight_gained_in_grams')


# In[62]:


resid_model3 = pred3-calories.Weight_gained_in_grams


# In[63]:


rmse_model3=np.sqrt(np.mean(resid_model3**2))


# In[64]:


rmse_model3


# aboves are the errors calculated from actual to predicted

# In[65]:


#Quadratic model

calories["Calories_consumed_sq"] = calories.Calories_consumed*calories.Calories_consumed


# In[66]:


print(calories)


# In[67]:


model_quad = smf.ols("Weight_gained_in_grams~(Calories_consumed+Calories_consumed_sq)",data=calories).fit()


# In[68]:


model_quad.params


# In[69]:


model_quad.summary()


# In[70]:


pred_quad = model_quad.predict(calories)


# In[71]:


#pred_quad = model_quad.predict(calories.iloc[:,1:3])


# In[72]:


pred_quad


# In[73]:


resid_error_model_quad=pred_quad-calories['Weight_gained_in_grams']


# In[74]:


rmse_model_quad=np.sqrt(np.mean(resid_error_model_quad**2))


# In[75]:


rmse_model_quad


# In[76]:


#                    r_sq      adj_r_sq     Aic     p_value             Rmse_value
#model               0.897     0.888        173     0.331               103.30
#model2              0.808     0.792        183     0.000               141.005
#model3              0.878     0.867        10.65   0.000               118.045
#model_quad          0.952     0.943        164.9   x=0.177,x^2=0.004   70.4075


# so from abobe we can conclude that our best model model3 or model_quad are good so we can will go for rmse_value in 
# which model_quad have small value

# In[77]:


student_resid = model_quad.resid_pearson 


# In[78]:


student_resid


# In[79]:


plt.plot(model_quad.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# in above graph if garph of standardized error is constant with observatin number is constant .so,on increasing no of observation our residual error will be constant so model_quad is fine according to our above graph

# In[80]:


# Predicted vs actual values
plt.scatter(x=pred_quad,y=calories.Weight_gained_in_grams);plt.xlabel("Predicted");plt.ylabel("Actual")


# in above graph prediction versus actual is linear graph so it shows that prediction which we have done is very good

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




