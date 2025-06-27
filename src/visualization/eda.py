import matplotlib.pyplot as plt

def scatter_featurs(df):
    df_0 = df[df['target'] == 0]
    plt.scatter(df_0['length_text'], df_0['length_title'], color='red', label='Target 0')

    # Plot points where target == 1 (green)
    df_1 = df[df['target'] == 1]
    plt.scatter(df_1['length_text'], df_1['length_title'], color='green', label='Target 1')

    # Add labels and legend
    plt.xlabel('Length of Text')
    plt.ylabel('Length of Title')
    plt.title('Text vs Title Length by Target')
    plt.legend()

    # Show plot
    plt.show()


    plt.scatter(df_0['avg_len_4plus'], df_0['text_title_length_relation'], color='red', label='Target 0')
    plt.scatter(df_1['avg_len_4plus'], df_1['text_title_length_relation'], color='green', label='Target 1')
    plt.xlabel('avg_len_4plus')
    plt.ylabel('text_title_length_relation')
    plt.title('text_title_length_relation vs avg_len_4plus  by Target')
    plt.legend()

    # Show plot
    plt.show()


    plt.scatter(df_0['avg_len_sentences'], df_0['number_word'], color='red', label='Target 0')
    plt.scatter(df_1['avg_len_sentences'], df_1['number_word'], color='green', label='Target 1')
    plt.xlabel('avg_len_sentences')
    plt.ylabel('number_word')
    plt.title('number_word vs avg_len_sentences  by Target')
    plt.legend()

    # Show plot
    plt.show()

    
    plt.scatter(df_0['vocabulary_richness'], df_0['num_sentences'], color='red', label='Target 0')
    plt.scatter(df_1['vocabulary_richness'], df_1['num_sentences'], color='green', label='Target 1')
    plt.xlabel('vocabulary_richness')
    plt.ylabel('num_sentences')
    plt.title('num_sentences vs vocabulary_richness  by Target')
    plt.legend()

    # Show plot
    plt.show()

    plt.scatter(df_0['avg_word_length'], df_0['num_unique_words'], color='red', label='Target 0')
    plt.scatter(df_1['avg_word_length'], df_1['num_unique_words'], color='green', label='Target 1')
    plt.xlabel('avg_word_length')
    plt.ylabel('num_unique_words')
    plt.title('num_unique_words vs vocabulavg_word_lengthary_richness  by Target')
    plt.legend()

    # Show plot
    plt.show()

    plt.scatter(df_0['exclamation_count'], df_0['punctuation_count'], color='red', label='Target 0')
    plt.scatter(df_1['exclamation_count'], df_1['punctuation_count'], color='green', label='Target 1')
    plt.xlabel('exclamation_count')
    plt.ylabel('punctuation_count')
    plt.title('punctuation_count vs exclamation_count  by Target')
    plt.legend()

    # Show plot
    plt.show()

    plt.scatter(df_0['num_uppercase_words'], df_0['question_count'], color='red', label='Target 0')
    plt.scatter(df_1['num_uppercase_words'], df_1['question_count'], color='green', label='Target 1')
    plt.xlabel('num_uppercase_words')
    plt.ylabel('question_count')
    plt.title('question_count vs num_uppercase_words  by Target')
    plt.legend()

    # Show plot
    plt.show()

    plt.scatter(df_0['contains_hyperlink'], df_0['title_word_overlap_ratio'], color='red', label='Target 0')
    plt.scatter(df_1['contains_hyperlink'], df_1['title_word_overlap_ratio'], color='green', label='Target 1')
    plt.xlabel('contains_hyperlink')
    plt.ylabel('title_word_overlap_ratio')
    plt.title('title_word_overlap_ratio vs contains_hyperlink  by Target')
    plt.legend()

    # Show plot
    plt.show()

    plt.scatter(df_0['readablility_text'], df_0['smog'], color='red', label='Target 0')
    plt.scatter(df_1['readablility_text'], df_1['smog'], color='green', label='Target 1')
    plt.xlabel('readablility_text')
    plt.ylabel('smog')
    plt.title('smog vs readablility_text  by Target')
    plt.legend()

    # Show plot
    plt.show()

    
    plt.scatter(df_0['difficalt_word'], df_0['smog'], color='red', label='Target 0')
    plt.scatter(df_1['difficalt_word'], df_1['smog'], color='green', label='Target 1')
    plt.xlabel('difficalt_word')
    plt.ylabel('smog')
    plt.title('smog vs difficalt_word  by Target')
    plt.legend()

    # Show plot
    plt.show()


    
    
    