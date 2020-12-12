#!/usr/bin/env python
# coding: utf-8

# In[2]:


# For reading data set
# importing necessary libraries
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


# 3)Years_experience -> Build a prediction model for Salary
# 
# Do the necessary transformations for input variables for getting better R^2 value for the model prepared.

# In[3]:


# reading a csv file using pandas library
data=pd.read_csv("Salary_Data.csv")


# In[4]:


data.columns


# In[5]:


plt.boxplot(data.Salary,0,"rs",0)


# In[6]:


plt.hist(data.Salary) #graph shows no normality


# In[7]:


plt.hist(data.YearsExperience) #shows that salry_hike is not normaly distributed


# In[8]:


plt.plot(data.YearsExperience,data.Salary,"bo");plt.xlabel("YearExperience");plt.ylabel("Salary")

#below graph shows that it have positive slope


# In[9]:


data.Salary.corr(data.YearsExperience) # # correlation value between X and Y


# In[10]:


#np.corrcoef(time_prediction.Delivery_time,time_prediction.Sorting_time)


# In[11]:


# For preparing linear regression model we need to import the statsmodels.formula.api
import statsmodels.formula.api as smf


# In[12]:


model1=smf.ols("Salary~YearsExperience",data=data).fit()


# In[13]:


model1.params


# In[14]:


model1.summary()


# In[15]:


pred1=model1.predict(data)


# In[16]:


pred1


# In[17]:


#pred1 = model1.predict(data.iloc[:,0]) # Predicted values of Salary using the model here we are predecting 
#salary using YearsExperience which is zeroth column thats why i passed [:,0]


# In[18]:


#pred1=model1.predict(pd.DataFrame(data['Salary_hike'])) #predicted values of Salary using the model
#here we r predicting Salary using YearsEperience


# In[19]:


resid_error1=pred1-data.Salary


# In[20]:


resid_error1


# In[21]:


model1.conf_int(0.05) # 95% confidence interval


# In[22]:


import matplotlib.pylab as plt


# In[23]:


plt.scatter(x=data['YearsExperience'],y=data['Salary'],color='red');plt.plot(data['YearsExperience'],pred1,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')


# In[24]:


pred1.corr(data.YearsExperience) # 0.99


# In[25]:


resid_error1=pred1-data.Salary


# In[26]:


resid_error1


# In[27]:


rmse_model1=np.sqrt(np.mean(resid_error1**2))


# In[28]:


rmse_model1


# In[29]:


# Transforming x variables for accuracy
model2 = smf.ols('Salary~np.log(YearsExperience)',data=data).fit()


# In[30]:


model2.params


# In[31]:


model2.summary()


# In[32]:


pred2=model2.predict(data)


# In[33]:


pred2


# In[34]:


resid_error_model2=pred2-data.Salary


# In[35]:


resid_error_model2


# In[36]:


rmse_model2=np.sqrt(np.mean(resid_error_model2**2))


# In[37]:


rmse_model2


# In[38]:


model2.conf_int(0.05) # 95% confidence interval


# In[39]:


plt.scatter(x=data['YearsExperience'],y=data['Salary'],color='red');plt.plot(data['YearsExperience'],pred2,color='black');plt.xlabel('YearsExperience');plt.ylabel('Salary')


# In[40]:


pred2.corr(data.Salary) # 0.92


# In[41]:


# Exponential transformation
model3 = smf.ols('np.log(Salary)~YearsExperience',data=data).fit()


# In[42]:


model3.params


# In[43]:


model3.summary()


# In[44]:


pred_log = model3.predict(pd.DataFrame(data['YearsExperience']))


# In[45]:


pred_log


# In[46]:


pred3=np.exp(pred_log)  # as we have used log(Churn_out_rata) in preparing model so we need to convert it back


# In[47]:


pred3


# In[48]:


pred3.corr(data.Salary)


# In[49]:


plt.scatter(x=data['YearsExperience'],y=data['Salary'],color='green');plt.plot(data.YearsExperience,np.exp(pred_log),color='blue');plt.xlabel('YearsExperience');plt.ylabel('Salary')


# In[50]:


resid_3 = pred3-data.Salary


# In[51]:


resid_3


# In[52]:


student_resid = pd.DataFrame(model3.resid_pearson) 


# In[53]:


student_resid


# In[54]:


rmse_model3=np.sqrt(np.mean(resid_3**2))


# In[55]:


rmse_model3


# In[56]:


# Quadratic model
data["YearsExperience_sq"] = data.YearsExperience*data.YearsExperience


# In[57]:


model4 = smf.ols("Salary~YearsExperience+YearsExperience_sq",data=data).fit()


# In[58]:


model4.params


# In[59]:


model4.summary()


# In[60]:


pred4 = model4.predict(data)


# In[61]:


pred4


# In[62]:


plt.scatter(data.YearsExperience,data.Salary,c="b");plt.plot(data.YearsExperience,pred4,"r")


# In[63]:


resid_4 = pred4-data.Salary


# In[64]:


resid_4


# In[65]:


rmse_model4=np.sqrt(np.mean(resid_4**2))


# In[66]:


rmse_model4


# In[67]:


# square root  transformation of Salary
model5 = smf.ols('np.sqrt(Salary)~YearsExperience',data=data).fit()


# In[68]:


model5.params


# In[69]:


model5.summary()


# In[70]:


pred_sq=model5.predict(pd.DataFrame(data['YearsExperience']))


# In[71]:


pred_sq


# In[72]:


pred5=np.square(pred_sq)


# In[73]:


pred5


# In[74]:


pred5.corr(data.YearsExperience)


# In[75]:


resid5=pred5-data.Salary


# In[76]:


resid5


# In[77]:


rmse_model5=np.sqrt(np.mean(resid5**2))


# In[78]:


rmse_model5


# In[79]:


# reciprocal transformation of Churn_out_rate 
model6 = smf.ols('np.reciprocal(Salary)~YearsExperience',data=data).fit()


# In[80]:


model6.params


# In[81]:


model6.summary()


# In[82]:


pred_reciprocal=model6.predict(pd.DataFrame(data['YearsExperience']))


# In[83]:


pred_reciprocal


# In[84]:


pred6=np.reciprocal(pred_reciprocal)


# In[85]:


pred6


# In[86]:


pred6.corr(data.Salary)


# In[87]:


resid6=pred6-data.Salary


# In[88]:


resid6


# In[89]:


rmse_model6=np.sqrt(np.mean(resid6**2))


# In[90]:


rmse_model6


# In[91]:


#log transformation of both independent and dependent variable
model7= smf.ols('np.log(Salary)~np.log(YearsExperience)',data=data).fit()


# In[92]:


model7.params


# In[93]:


model7.summary()


# In[94]:


pred_lg=model7.predict(pd.DataFrame(data['YearsExperience']))


# In[95]:


pred_lg


# In[96]:


pred7=np.exp(pred_lg)


# In[97]:


pred7


# In[98]:


pred7.corr(data.Salary)


# In[99]:


resid7=pred7-data.Salary


# In[100]:


rmse_model7=np.sqrt(np.mean(resid7**2))


# In[101]:


rmse_model7


# In[102]:


#                    r_sq    adj_r_sq     AIC      P_value        rmse_model
#    
#    model1         0.957    0.955        606.9    0.00            5392
#    model2         0.854    0.854        643.5    0.00            10302
#    model3         0.932    0.932       -52.37    0.112           7213
#    model4         0.963     0.966      604.00 (x=1200,x^2=-8.801)5192.1831
#    model5         0.882    0.878        258.3   -8.801           9258.43
#    model6         0.724    0.714      -677.1     -1.72e-07       24171.18
#    model7         0.905    0.902       -42.42    0.00            7219


#from above value if we compare then model3 is the best model among all models


# In[103]:


# so we will consider the model having highest R-Squared value which is the log transformation of both independent as well as 
#dependent variables - model3
# getting residuals of the entire data set
student_resid1 = pd.DataFrame(model3.resid_pearson)


# In[104]:


student_resid1


# In[105]:


plt.plot(model3.resid_pearson,'o');plt.axhline(y=0,color='green');plt.xlabel("Observation Number");plt.ylabel("Standardized Residual")


# In[106]:


# Predicted vs actual values
plt.scatter(x=pred3,y=data.Salary);plt.xlabel("Predicted");plt.ylabel("Actual")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




