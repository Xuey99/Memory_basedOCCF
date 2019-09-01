# Recommendation with Implicit Feekback
## Memory_basedOCCF   
## Collaborative Filtering with implicit feedback 

Like memory-base CF, the algorithm intorduced before, Memorybased-OCCF is also a method based on the collective wisdom. 😀

Three kinds of OCCF correspondingly based on items, users,  both items and users.

The algorithm is described as following steps:
- ① compute the similarity between every two items or users, and normalize them 
- ② select the top-K most nearest neighborhoods w.r.t the similarity measurement.
- ③ utilize predict rules to predict the rating for the missing value in the martix and rank them to recommend the top k items for user u

