#!/usr/bin/env python
# coding: utf-8

# 2) Delivery_time -> Predict delivery time using sorting time 
# 
# Do the necessary transformations for input variables for getting better R^2 value for the model prepared.
# 

# In[2]:


import pandas as pd ## importing necessary libraries pandas


# In[3]:


import numpy as np


# In[4]:


import matplotlib.pyplot as plt


# reading a csv file using pandas library

# In[5]:


time_prediction=pd.read_csv(r"D:\excel_R_lms\simple linear assignment excelR\delivery_time.csv")


# In[6]:


time_prediction.columns


# In[7]:


plt.boxplot(time_prediction.Delivery_time,0,"rs",0)


# In[8]:


plt.hist(time_prediction.Delivery_time)


# In[9]:


plt.hist(time_prediction.Sorting_time)


# In[10]:


plt.boxplot(time_prediction.Sorting_time,0,"rs",0)


# in above all grphs it shows that both X and Y are not normalized

# In[11]:


plt.plot(time_prediction.Sorting_time,time_prediction.Delivery_time,"bo");plt.xlabel("Sorting_time");plt.ylabel("Delivery_time")


# it shows that less corelation b/w between Delivery_time and Sorting_time

# In[12]:


time_prediction.Delivery_time.corr(time_prediction.Sorting_time) # # correlation value between X and Y


# In[13]:


#np.corrcoef(time_prediction.Delivery_time,time_prediction.Sorting_time)


# In[14]:


# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf


# In[15]:


model=smf.ols("Delivery_time~Sorting_time",data=time_prediction).fit()


# In[16]:


# For getting coefficients of the varibles used in equation
model.params


# In[17]:


# P-values for the variables and R-squared value for prepared model
model.summary()


# In[18]:


pred=model.predict(time_prediction)


# In[19]:


pred


# In[20]:


#pred = model.predict(time_prediction.iloc[:,1]) # Predicted values of Delivery_time using the model here we are predecting 
#Delivery_time using Sorting_time which is zeroth column thats why i passed [:,1]


# In[21]:


#pred=model.predict(pd.DataFrame(time_prediction['Sorting_time'])) #predicted values of Delivery_time using the model
#here we r predicting Sorting_time using Delivery_time


# In[22]:


resid_error=pred-time_prediction.Delivery_time


# In[23]:


rmse_model=np.sqrt(np.mean(resid_error**2))


# In[24]:


rmse_model


# In[25]:


time_prediction.Delivery_time.corr(time_prediction.Sorting_time) # # correlation value between X and Y


# In[26]:


model.conf_int(0.05) # 95% confidence interval


# In[27]:


import matplotlib.pylab as plt


# In[28]:


plt.scatter(x=time_prediction['Sorting_time'],y=time_prediction['Delivery_time'],color='red');plt.plot(time_prediction['Sorting_time'],pred,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time')


# In[29]:


pred.corr(time_prediction.Delivery_time) # 0.81


# In[30]:


# Transforming variables for accuracy
model2 = smf.ols('Delivery_time~np.log(Sorting_time)',data=time_prediction).fit()


# In[31]:


model2.params


# In[32]:


model2.summary()


# In[33]:


pred2=model2.predict(time_prediction)


# In[34]:


pred2


# In[35]:


#pred2 = model2.predict(time_prediction.iloc[:,1]) # Predicted values of Delivery_time using the model here we are predecting 
#Delivery_time using Sorting_time which is zeroth column thats why i passed [:,1]


# In[36]:


#pred2=model2.predict(pd.DataFrame(time_prediction['Sorting_time'])) #predicted values of Delivery_time using the model
#here we r predicting Sorting_time using Delivery_time


# In[37]:


resid_error_model2=pred2-time_prediction.Delivery_time


# In[38]:


resid_error_model2


# In[39]:


rmse_model2=np.sqrt(np.mean(resid_error_model2**2))


# In[40]:


rmse_model2


# In[41]:


model2.conf_int(0.05) # 95% confidence interval


# In[42]:


plt.scatter(x=time_prediction['Sorting_time'],y=time_prediction['Delivery_time'],color='red');plt.plot(time_prediction['Sorting_time'],pred,color='black');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time')


# In[43]:


pred2.corr(time_prediction.Delivery_time) # 0.83


# In[44]:


# Exponential transformation
model3 = smf.ols('np.log(Delivery_time)~Sorting_time',data=time_prediction).fit()


# In[45]:


model3.params


# In[46]:


model3.summary()


# In[47]:


pred_log = model3.predict(pd.DataFrame(time_prediction['Sorting_time']))


# In[48]:


pred_log


# In[49]:


pred3=np.exp(pred_log)  # as we have used log(Delivery_time) in preparing model so we need to convert it back


# In[50]:


pred3


# In[51]:


pred3.corr(time_prediction.Delivery_time)


# In[52]:


plt.scatter(x=time_prediction['Sorting_time'],y=time_prediction['Delivery_time'],color='green');plt.plot(time_prediction.Sorting_time,np.exp(pred_log),color='blue');plt.xlabel('Sorting_time');plt.ylabel('Delivery_time')


# In[53]:


resid_3 = pred3-time_prediction.Delivery_time


# In[54]:


resid_3


# In[55]:


student_resid = pd.DataFrame(model3.resid_pearson) 


# In[56]:


student_resid


# In[57]:


rmse_model3=np.sqrt(np.mean(resid_3**2))


# In[58]:


rmse_model3


# In[59]:


plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# In[60]:


# Quadratic model
time_prediction["Sorting_time_sq"] = time_prediction.Sorting_time*time_prediction.Sorting_time


# In[61]:


model_quad = smf.ols("Delivery_time~Sorting_time+Sorting_time_sq",data=time_prediction).fit()


# In[62]:


model_quad.params


# In[63]:


model_quad.summary()


# In[64]:


pred_quad = model_quad.predict(time_prediction)


# In[65]:


pred_quad


# In[66]:


plt.scatter(time_prediction.Sorting_time,time_prediction.Delivery_time,c="b");plt.plot(time_prediction.Sorting_time,pred_quad,"r")


# In[67]:


resid_quad = pred_quad-time_prediction.Delivery_time


# In[68]:


resid_quad


# In[69]:


rmse_model_quad=np.sqrt(np.mean(resid_quad**2))


# In[70]:


rmse_model_quad


# In[71]:


# square root  transformation of Delivery_time
model4 = smf.ols('np.sqrt(Delivery_time)~Sorting_time',data=time_prediction).fit()


# In[72]:


model4.params


# In[73]:


model4.summary()


# In[74]:


pred_sq=model4.predict(pd.DataFrame(time_prediction['Sorting_time']))


# In[75]:


pred_sq


# In[76]:


pred4=np.square(pred_sq)


# In[77]:


pred4


# In[78]:


pred4.corr(time_prediction.Delivery_time)


# In[79]:


resid4=pred4-time_prediction.Delivery_time


# In[80]:


resid4


# In[81]:


rmse_model4=np.sqrt(np.mean(resid4**2))


# In[82]:


rmse_model4


# In[83]:


# reciprocal transformation of Delivery_time 
model5 = smf.ols('np.reciprocal(Delivery_time)~Sorting_time',data=time_prediction).fit()


# In[84]:


model5.params


# In[85]:


model5.summary()


# In[87]:


pred_reciprocal=model5.predict(pd.DataFrame(time_prediction['Sorting_time']))


# In[88]:


pred_reciprocal


# In[89]:


pred5=np.reciprocal(pred_reciprocal)


# In[90]:


pred5


# In[91]:


pred5.corr(time_prediction.Delivery_time)


# In[92]:


resid5=pred5-time_prediction.Delivery_time


# In[93]:


resid5


# In[94]:


rmse_model5=np.sqrt(np.mean(resid5**2))


# In[95]:


rmse_model5


# In[97]:


#log transformation of both independent and dependent variable
model6 = smf.ols('np.log(Delivery_time)~np.log(Sorting_time)',data=time_prediction).fit()


# In[98]:


model6.params


# In[99]:


model6.summary()


# In[100]:


pred_lg=model6.predict(pd.DataFrame(time_prediction['Sorting_time']))


# In[101]:


pred_lg


# In[102]:


pred6=np.exp(pred_lg)


# In[103]:


pred6


# In[104]:


pred6.corr(time_prediction.Delivery_time)


# In[105]:


resid6=pred6-time_prediction.Delivery_time


# In[106]:


resid6


# In[107]:


rmse_model6=np.sqrt(np.mean(resid6**2))


# In[108]:


rmse_model6


# In[109]:


#                    r_sq    adj_r_sq     AIC      P_value        rmse_model
#    
#    model1          0.682   0.666        77.68    1.109           2.79
#    model2         0.695    0.679        105.8    6.130           2.7331
#    model3         0.711    0.69       -11.58     0.073           2.940
#    quad_model     0.673    0.637       109.3  (x=0.428,x^2=0.00) 2.8301
#    model4         0.633    0.613       21.87     0.000           3.09
#    model5         0.68     0.665      -120.9    -0.010           3.3972
#    model6         0.772    0.760      -16.58     0.000           2.7458


#from above value if we compare then model 6 is the best model among all models


# In[113]:


# so we will consider the model having highest R-Squared value which is the log transformation of both independent as well as 
#dependent variables - model3
# getting residuals of the entire data set
student_resid = pd.DataFrame(model6.resid_pearson) 


# In[114]:


student_resid


# In[115]:


plt.plot(model6.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# above graph shaows that residual error will remain constatnt on inreasing the no of obeservation 

# In[116]:


# Predicted vs actual values
plt.scatter(x=pred6,y=time_prediction.Delivery_time);plt.xlabel("Predicted");plt.ylabel("Actual")


# In[ ]:




