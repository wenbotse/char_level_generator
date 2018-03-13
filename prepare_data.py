import pandas as pd
df=pd.read_csv('../../dataset/business.csv')
df[['name','categories']].head(10)

df.head()
df.shape
df2=pd.read_csv('../../dataset/review.csv')
df2.shape

df2[['text','stars']].head()

five_star=df2[df2['stars']==5]
five_star.shape

restaurants = df[df['categories'].str.contains('Restaurants')]

restaurants.shape
restaurants_clean=restaurants[['business_id','name']]
restaurants_clean.head()
restaurants_clean.shape
combo=pd.merge(restaurants_clean, five_star, on='business_id')
combo.shape
combo.head()
rnn_fivestar_reviews_only=combo[['text']]
rnn_fivestar_reviews_only.head()
rnn_fivestar_reviews_only.shape
rnn_fivestar_reviews_only=rnn_fivestar_reviews_only.replace({r'\n+': ''}, regex=True)
final=rnn_fivestar_reviews_only.drop_duplicates()
final.shape
final['text'][98836]
#Add start and end of review with <SOR> and <EOR> to each review
final['text']='<SOR>' + final['text'].astype(str)+'<EOR>'
final.head()
final['text'][98836]
final.loc[3][0]
#create a csv of this for future use
filename='../../dataset/five_star_restaurants_reviews_only.csv'
final.to_csv(filename, index=False, encoding='utf-8')
#full text file of all reviews to train on
filename='../../dataset/five_star_text.txt'
final.to_csv(filename, header=None, index=None, sep=' ')

