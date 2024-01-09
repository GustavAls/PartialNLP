
import os

from os import path
from wordcloud import WordCloud

# get data directory (using getcwd() is needed to support running example in generated IPython notebook)
d = path.dirname(__file__) if "__file__" in locals() else os.getcwd()

# Read the whole text.
text = open(r"C:\Users\45292\Documents\Master\NLP\SST2\Figures\Examples\Data\wrongly_labeled_data_cls_one.txt").read()

# Generate a word cloud image
wordcloud = WordCloud().generate(text)

# Display the generated image:
# the matplotlib way:
import matplotlib.pyplot as plt
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")

# lower max_font_size
wordcloud = WordCloud(max_font_size=40).generate(text)

fig, ax = plt.subplots(1, 1)
ax.imshow(wordcloud, interpolation="bilinear")
ax.axis("off")

save_path = r"C:\Users\45292\Documents\Master\NLP\SST2\Figures\Examples\Wordcloud\wrongly_labeled_class_one.pdf"
fig.tight_layout()
fig.savefig(save_path, format = 'pdf')
