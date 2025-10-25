import joblib

vectorizer = joblib.load("vectorizer.pkl")
model = joblib.load("linear_model.pkl")

new_essay = """International sports events require the most well-trained athletes for each country, in order to achieve this goal countries make an effort to build infrastructure designed to train top athletes. Although this policy can indeed make fewer sports facilities for ordinary people, investing in the best athletes is vital to develop competitive sports performances in each country.
On the one hand, building specific infrastructure for the best athletes is crucial in order to get better results at international sports events such as The Olympics or the World Cup. The importance of getting better results is that it creates awareness of the importance of sports in society and motivates more people to do a sport. In this way, investing in these developments can help countries to develop an integral sport policy that can benefit everyone.
On the other hand, one can argue that a negative effect could be that less infrastructure is built for the rest of the people. However, people who practice a sport in their daily life do not necessarily need some facilities to do sports. For example, people often use public spaces to do sports such as running or doing yoga at the nearest park to their home. So, for people who is not top athletes there could be some alternatives for sports facility that ,is not the case for training top athletes.
To sum up, I strongly believe countries should invest in specialised infrastructure for their best athletes because in the long term is going to generate more motivation to do sports, to invest in sports at schools and therefore to build more sports infrastructure for everyone."""

new_vec = vectorizer.transform([new_essay])
predicted_grade = model.predict(new_vec)
print(f"The predicted grade is: {predicted_grade[0]:.2f}")
