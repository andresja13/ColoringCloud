# adapted from 
# https://amueller.github.io/word_cloud/auto_examples/colored_by_group.html
from wordcloud import WordCloud, get_single_color_func 
import matplotlib.pyplot as plt 
from UsefulSentimentAnalysis import lemmatized_tweets, positive_words, negative_words, pos_word_color, neg_word_color

class SimpleGroupedColorFunc(object):
    ''' create a color function opject which assigns exact colors 
    to certain words based on the color to words mapping'''

    # color_to_words: dict(str -> list(str))
    # dict that maps a color to the list of words 

    # default_color: str 
    # the color that will be assigned to any word that's not a member of any value from color_to_words

    def __init__(self, color_to_words, default_color):
        self.word_to_color = {word: color for (color, words) in color_to_words.items() for word in words}
        self.default_color = default_color
    
    def __call__(self, word, **kwargs):
        return self.word_to_color.get(word, self.default_color)
    
class GroupedColorFunc(object):
    ''' create a color function object which assigns DIFFERENT SHADES of 
    specified colors to certain words based on the color to words mapping '''

    # color_to_words: dict(str -> list(str))
    # dict that maps a color to the list of words

    # default_color: str 
    # the color that will be assigned to any word that's not a member of any value from color_to_word

    def __init__(self, color_to_words, default_color):
        self.color_funct_to_words = [(get_single_color_func(color), set(words)) for (color, words) in color_to_words.items()]
        self.default_color_func = get_single_color_func(default_color)
    
    def get_color_func(self, word):
        ''' returns a single_color_func associated with the word '''
        try: 
            color_func = next(color_func for (color_func, words) in self.color_funct_to_words if word in words)
        except StopIteration:
            color_func = self.default_color_func
        return color_func 
    
    def __call__(self, word, **kwargs):
        return self.get_color_func(word)(word, **kwargs)

# create the wordcloud 

# this method of using all the words in all the tweets works
lem_list = [' '.join(ele) for ele in lemmatized_tweets]
lem_list = ', '.join(lem_list)

# this method of using only the words that are in the predicted words set works
# but may be less accurate because each one is only counted once
#lem_list = list(predicted_words)
#lem_list = ', '.join(lem_list)


wc = WordCloud(collocations=False, max_words=50).generate(lem_list.lower())

color_to_words = {
    'green': pos_word_color["green"],
    'red': neg_word_color["red"]
}


default_color = 'grey'

'''using either of these works'''
# create a color function with single tone 
grouped_color_func = SimpleGroupedColorFunc(color_to_words, default_color)
# create a color function with multiple tones 
#grouped_color_func = GroupedColorFunc(color_to_words, default_color)

# apply the color function 
wc.recolor(color_func=grouped_color_func)

# plot 
plt.figure()
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()