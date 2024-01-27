{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "cc972ecd",
   "metadata": {},
   "outputs": [],
   "source": [
    "class Sentiment:\n",
    "    NEGATIVE = \"NEGATIVE\"\n",
    "    NEUTRAL = \"NEUTRAL\"\n",
    "    POSITIVE = \"POSITIVE\"\n",
    "\n",
    "class Review:\n",
    "    def __init__(self, text, score):\n",
    "        self.text = text\n",
    "        self.score = score\n",
    "        self.sentiment = self.get_sentiment()\n",
    "\n",
    "    def get_sentiment(self):\n",
    "        if self.score <= 2:\n",
    "            return Sentiment.NEGATIVE\n",
    "        elif self.score == 3:\n",
    "            return Sentiment.NEUTRAL\n",
    "        else:  # Score of 4 or 5\n",
    "            return Sentiment.POSITIVE\n",
    "        \n",
    "class ReviewContainer:\n",
    "    def __init__(self, reviews):\n",
    "        self.reviews = reviews\n",
    "    def evenly_distribute(self):\n",
    "        negative = list(filter (lambda x: x.sentiment == Sentiment.NEGATIVE, self.reviews))\n",
    "        positive = list(filter (lambda x: x.sentiment == Sentiment.POSITIVE, self.reviews))\n",
    "        print(negative[0].text)\n",
    "        print(len(negative))\n",
    "        print(len(positive))\n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88e1a9a4",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "id": "78a9ce92",
   "metadata": {},
   "source": [
    "### Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 65,
   "id": "31a67e2e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'I hoped for Mia to have some peace in this book, but her story is so real and raw.  Broken World was so touching and emotional because you go from Mia\\'s trauma to her trying to cope.  I love the way the story displays how there is no \"just bouncing back\" from being sexually assaulted.  Mia showed us how those demons come for you every day and how sometimes they best you. I was so in the moment with Broken World and hurt with Mia because she was surrounded by people but so alone and I understood her feelings.  I found myself wishing I could give her some of my courage and strength or even just to be there for her.  Thank you Lizzy for putting a great character\\'s voice on a strong subject and making it so that other peoples story may be heard through Mia\\'s.'"
      ]
     },
     "execution_count": 65,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "import json\n",
    "\n",
    "file_name = './model/books_small_10000.json'\n",
    "\n",
    "\n",
    "reviews = []  # empty list\n",
    "\n",
    "with open(file_name) as f:\n",
    "    for line in f:\n",
    "        review = json.loads(line)\n",
    "        reviews.append(Review(review['reviewText'], review['overall']))\n",
    "\n",
    "reviews[5].text\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "40da0ff6",
   "metadata": {},
   "source": [
    "### Prep Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "ac6d9491",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "It was just one of those books that never went anywhere. I like books that get your attention in the beginning and not drag out until a quarter way through. I decided to give it an early death - delete!\n",
      "436\n",
      "5611\n"
     ]
    }
   ],
   "source": [
    "from sklearn.model_selection import train_test_split\n",
    "\n",
    "training,test = train_test_split(reviews, test_size =0.33, random_state=42)\n",
    "\n",
    "cont = ReviewContainer(training)\n",
    "cont.evenly_distribute()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "e1ebd349",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "'POSITIVE'"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_x = [x.text for x in training]\n",
    "train_y = [x.sentiment for x in training]\n",
    "\n",
    "test_x = [x.text for x in test]\n",
    "test_y = [x.sentiment for x in test]\n",
    "\n",
    "train_x[0]\n",
    "train_y[0]"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4fe221a5",
   "metadata": {},
   "source": [
    "#### Bag of words vectorization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "e09ae953",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Olivia Hampton arrives at the Dunraven family home as cataloger of their extensive library. What she doesn't expect is a broken carriage wheel on the way. Nor a young girl whose mind is clearly gone, an old man in need of care himself (and doesn&#8217;t quite seem all there in Olivia&#8217;s opinion). Furthermore, Marion Dunraven, the only sane one of the bunch and the one Olivia is inexplicable drawn to, seems captive to everyone in the dusty old house. More importantly, she doesn't expect to fall in love with Dunraven's daughter Marion.Can Olivia truly believe the stories of sadness and death that surround the house, or are they all just local neighborhood rumor?Was that carriage trouble just a coincidence or a supernatural sign to stay away? If she remains, will the Castle&#8217;s dark shadows take Olivia down with them or will she and Marion long enough to declare their love?Patty G. Henderson has created an atmospheric and intriguing story in her Gothic tale. I found this to be an enjoyable read, even if it isn&#8217;t my usual preferred genre. I think, with this tale, I got hooked on the old Gothic romantic style. So I think fans of the genre (and of lesbian romances) will enjoy it.\n",
      "[[0 0 0 ... 0 0 0]]\n"
     ]
    }
   ],
   "source": [
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "\n",
    "vectorizer = CountVectorizer()\n",
    "train_x_vectors = vectorizer.fit_transform(train_x)\n",
    "\n",
    "test_x_vectors = vectorizer.transform(test_x)\n",
    "\n",
    "print(train_x[0])\n",
    "print(train_x_vectors[0].toarray())\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4117353e",
   "metadata": {},
   "source": [
    "## Classification"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "57704e84",
   "metadata": {},
   "source": [
    "#### Linear SVM"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 70,
   "id": "2cc1b47b",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['POSITIVE'], dtype='<U8')"
      ]
     },
     "execution_count": 70,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn import svm\n",
    "\n",
    "clf_svm = svm.SVC(kernel = 'linear')\n",
    "\n",
    "clf_svm.fit(train_x_vectors, train_y)\n",
    "\n",
    "test_x[0]\n",
    "\n",
    "clf_svm.predict(test_x_vectors[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "93d06be9",
   "metadata": {},
   "source": [
    "#### Decision Tree"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 71,
   "id": "ac4f804e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['POSITIVE'], dtype='<U8')"
      ]
     },
     "execution_count": 71,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.tree import DecisionTreeClassifier\n",
    "\n",
    "clf_dec = DecisionTreeClassifier()\n",
    "\n",
    "clf_dec.fit(train_x_vectors, train_y)\n",
    "\n",
    "clf_dec.predict(test_x_vectors[0])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "94db8231",
   "metadata": {},
   "source": [
    "#### Naive Bayes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 72,
   "id": "ef803198",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['POSITIVE'], dtype='<U8')"
      ]
     },
     "execution_count": 72,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.naive_bayes import GaussianNB\n",
    "\n",
    "clf_gnb = DecisionTreeClassifier()\n",
    "\n",
    "clf_gnb.fit(train_x_vectors, train_y)\n",
    "\n",
    "clf_gnb.predict(test_x_vectors[0])\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0c1130ce",
   "metadata": {},
   "source": [
    "#### Logistic Regression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "e2d59803",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "/Users/praiseayoade/anaconda3/lib/python3.11/site-packages/sklearn/linear_model/_logistic.py:460: ConvergenceWarning: lbfgs failed to converge (status=1):\n",
      "STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.\n",
      "\n",
      "Increase the number of iterations (max_iter) or scale the data as shown in:\n",
      "    https://scikit-learn.org/stable/modules/preprocessing.html\n",
      "Please also refer to the documentation for alternative solver options:\n",
      "    https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression\n",
      "  n_iter_i = _check_optimize_result(\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['POSITIVE'], dtype='<U8')"
      ]
     },
     "execution_count": 73,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.linear_model import LogisticRegression\n",
    "\n",
    "clf_lg = LogisticRegression()\n",
    "\n",
    "clf_lg.fit(train_x_vectors, train_y)\n",
    "\n",
    "clf_lg.predict(test_x_vectors[0])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "0ef7196c",
   "metadata": {},
   "source": [
    "## Evaluation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "649619a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8124242424242424\n",
      "0.7736363636363637\n",
      "0.7672727272727272\n",
      "0.8418181818181818\n"
     ]
    }
   ],
   "source": [
    "# Mean Accuracy \n",
    "print (clf_svm.score(test_x_vectors, test_y))\n",
    "print (clf_dec.score(test_x_vectors, test_y))\n",
    "print(clf_gnb.score(test_x_vectors, test_y)) \n",
    "print(clf_lg.score(test_x_vectors, test_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "27b20ed9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0.8741769 , 0.15151515, 0.16020672])"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# F1 Scores\n",
    "from sklearn.metrics import f1_score\n",
    "\n",
    "f1_score(test_y, clf_svm.predict(test_x_vectors), average =None, labels = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])\n",
    "f1_score(test_y, clf_gnb.predict(test_x_vectors), average =None, labels = [Sentiment.POSITIVE, Sentiment.NEUTRAL, Sentiment.NEGATIVE])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "1ec82bec",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array(['POSITIVE', 'NEGATIVE', 'NEGATIVE'], dtype='<U8')"
      ]
     },
     "execution_count": 76,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "test_set = ['very fun', \"bad book do not buy\", 'horrible waste of time']\n",
    "new_test = vectorizer.transform(test_set)\n",
    "\n",
    "clf_svm.predict(new_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "4908a9f3",
   "metadata": {},
   "source": [
    "## Turning Model with Grid search"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 77,
   "id": "99a1758e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<style>#sk-container-id-4 {color: black;}#sk-container-id-4 pre{padding: 0;}#sk-container-id-4 div.sk-toggleable {background-color: white;}#sk-container-id-4 label.sk-toggleable__label {cursor: pointer;display: block;width: 100%;margin-bottom: 0;padding: 0.3em;box-sizing: border-box;text-align: center;}#sk-container-id-4 label.sk-toggleable__label-arrow:before {content: \"▸\";float: left;margin-right: 0.25em;color: #696969;}#sk-container-id-4 label.sk-toggleable__label-arrow:hover:before {color: black;}#sk-container-id-4 div.sk-estimator:hover label.sk-toggleable__label-arrow:before {color: black;}#sk-container-id-4 div.sk-toggleable__content {max-height: 0;max-width: 0;overflow: hidden;text-align: left;background-color: #f0f8ff;}#sk-container-id-4 div.sk-toggleable__content pre {margin: 0.2em;color: black;border-radius: 0.25em;background-color: #f0f8ff;}#sk-container-id-4 input.sk-toggleable__control:checked~div.sk-toggleable__content {max-height: 200px;max-width: 100%;overflow: auto;}#sk-container-id-4 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {content: \"▾\";}#sk-container-id-4 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 input.sk-hidden--visually {border: 0;clip: rect(1px 1px 1px 1px);clip: rect(1px, 1px, 1px, 1px);height: 1px;margin: -1px;overflow: hidden;padding: 0;position: absolute;width: 1px;}#sk-container-id-4 div.sk-estimator {font-family: monospace;background-color: #f0f8ff;border: 1px dotted black;border-radius: 0.25em;box-sizing: border-box;margin-bottom: 0.5em;}#sk-container-id-4 div.sk-estimator:hover {background-color: #d4ebff;}#sk-container-id-4 div.sk-parallel-item::after {content: \"\";width: 100%;border-bottom: 1px solid gray;flex-grow: 1;}#sk-container-id-4 div.sk-label:hover label.sk-toggleable__label {background-color: #d4ebff;}#sk-container-id-4 div.sk-serial::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: 0;}#sk-container-id-4 div.sk-serial {display: flex;flex-direction: column;align-items: center;background-color: white;padding-right: 0.2em;padding-left: 0.2em;position: relative;}#sk-container-id-4 div.sk-item {position: relative;z-index: 1;}#sk-container-id-4 div.sk-parallel {display: flex;align-items: stretch;justify-content: center;background-color: white;position: relative;}#sk-container-id-4 div.sk-item::before, #sk-container-id-4 div.sk-parallel-item::before {content: \"\";position: absolute;border-left: 1px solid gray;box-sizing: border-box;top: 0;bottom: 0;left: 50%;z-index: -1;}#sk-container-id-4 div.sk-parallel-item {display: flex;flex-direction: column;z-index: 1;position: relative;background-color: white;}#sk-container-id-4 div.sk-parallel-item:first-child::after {align-self: flex-end;width: 50%;}#sk-container-id-4 div.sk-parallel-item:last-child::after {align-self: flex-start;width: 50%;}#sk-container-id-4 div.sk-parallel-item:only-child::after {width: 0;}#sk-container-id-4 div.sk-dashed-wrapped {border: 1px dashed gray;margin: 0 0.4em 0.5em 0.4em;box-sizing: border-box;padding-bottom: 0.4em;background-color: white;}#sk-container-id-4 div.sk-label label {font-family: monospace;font-weight: bold;display: inline-block;line-height: 1.2em;}#sk-container-id-4 div.sk-label-container {text-align: center;}#sk-container-id-4 div.sk-container {/* jupyter's `normalize.less` sets `[hidden] { display: none; }` but bootstrap.min.css set `[hidden] { display: none !important; }` so we also need the `!important` here to be able to override the default hidden behavior on the sphinx rendered scikit-learn.org. See: https://github.com/scikit-learn/scikit-learn/issues/21755 */display: inline-block !important;position: relative;}#sk-container-id-4 div.sk-text-repr-fallback {display: none;}</style><div id=\"sk-container-id-4\" class=\"sk-top-container\"><div class=\"sk-text-repr-fallback\"><pre>GridSearchCV(cv=5, estimator=SVC(),\n",
       "             param_grid={&#x27;C&#x27;: (1, 4, 8, 16, 32), &#x27;kernel&#x27;: (&#x27;linear&#x27;, &#x27;rbf&#x27;)})</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class=\"sk-container\" hidden><div class=\"sk-item sk-dashed-wrapped\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-10\" type=\"checkbox\" ><label for=\"sk-estimator-id-10\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">GridSearchCV</label><div class=\"sk-toggleable__content\"><pre>GridSearchCV(cv=5, estimator=SVC(),\n",
       "             param_grid={&#x27;C&#x27;: (1, 4, 8, 16, 32), &#x27;kernel&#x27;: (&#x27;linear&#x27;, &#x27;rbf&#x27;)})</pre></div></div></div><div class=\"sk-parallel\"><div class=\"sk-parallel-item\"><div class=\"sk-item\"><div class=\"sk-label-container\"><div class=\"sk-label sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-11\" type=\"checkbox\" ><label for=\"sk-estimator-id-11\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">estimator: SVC</label><div class=\"sk-toggleable__content\"><pre>SVC()</pre></div></div></div><div class=\"sk-serial\"><div class=\"sk-item\"><div class=\"sk-estimator sk-toggleable\"><input class=\"sk-toggleable__control sk-hidden--visually\" id=\"sk-estimator-id-12\" type=\"checkbox\" ><label for=\"sk-estimator-id-12\" class=\"sk-toggleable__label sk-toggleable__label-arrow\">SVC</label><div class=\"sk-toggleable__content\"><pre>SVC()</pre></div></div></div></div></div></div></div></div></div></div>"
      ],
      "text/plain": [
       "GridSearchCV(cv=5, estimator=SVC(),\n",
       "             param_grid={'C': (1, 4, 8, 16, 32), 'kernel': ('linear', 'rbf')})"
      ]
     },
     "execution_count": 77,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.model_selection import GridSearchCV\n",
    "\n",
    "parameters = {'kernel': ('linear', 'rbf'), 'C' : (1,4,8,16,32)}\n",
    "\n",
    "svc =svm.SVC()\n",
    "clf = GridSearchCV(svc, parameters, cv=5)\n",
    "clf.fit(train_x_vectors, train_y)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 78,
   "id": "83612cd4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.8124242424242424\n"
     ]
    }
   ],
   "source": [
    "print (clf_svm.score(test_x_vectors, test_y))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 79,
   "id": "917022b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "# good! good"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "006b0683",
   "metadata": {},
   "source": [
    "## Saving Model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 80,
   "id": "6d06a441",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pickle\n",
    "\n",
    "with open ('./model/sentiment_analyis.pkl', 'wb') as f:\n",
    "    pickle.dump(clf, f)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "61320f93",
   "metadata": {},
   "source": [
    "## Load model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 81,
   "id": "40deb261",
   "metadata": {},
   "outputs": [],
   "source": [
    "with open ('./model/sentiment_analyis.pkl', 'rb') as f:\n",
    "    loaded_clf = pickle.load(f)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "id": "694cf550",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Yet another wonderful book by Elle. I really hope there are more in this series. They're lol funny, have you in tears, and have your nerves going all in one. I normally don't write reviews (I always rare, which Elle is almost always a five star in my opinion), but as other reviewers said, this was the best out of the three. And there could be so much more to their stories....I want to know more!!! Keep up the wonderful writing Elle. I love your work.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "array(['POSITIVE'], dtype='<U8')"
      ]
     },
     "execution_count": 82,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(test_x[1])\n",
    "\n",
    "loaded_clf.predict(test_x_vectors[0])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37fa4319",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
