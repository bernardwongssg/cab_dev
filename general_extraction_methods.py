import os
import pandas as pd
import pypdf
import shutil
from PIL import Image
import numpy as np
from pdf2image import convert_from_path
from matplotlib import pyplot as plt
import cv2
import pytesseract
import fitz
from sqlalchemy import create_engine 
from dateutil.parser import parse
import string
from itertools import compress

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import classification_report

def prediction_precleaning(data_df, metric_col_name):
    '''
    implements cleaning rules before categorizing. note that this should be applied to both the model and the testing data
    '''
    print('pre-cleaning data...')
    cleaned_data_df = data_df.copy()

    ########### CLEANING RULES, SHOULD BE THE SAME AS MODEL ###############################
    # preprocessed cleaning
    print(' - dropping null values')
    classifier_df = cleaned_data_df.dropna()
    # lowercase, remove punctuation, remove numbers
    print(' - lowercasing, removing punctuation, removing numbers')
    classifier_df['preprocessed_metric'] = classifier_df[metric_col_name].apply(lambda x: x.lower().translate(str.maketrans('', '', string.punctuation)).translate(str.maketrans('', '', string.digits)))
    # remove months 
    '''
    print (' - remove months')
    classifier_df['preprocessed_metric'] = classifier_df['preprocessed_metric'].apply(lambda cleaned_metric: ' '.join(list(compress(cleaned_metric.split(), [is_date(x) == False for x in cleaned_metric.split()]))))
    '''
    # remove empty strings 
    print(' - remove empty strings')
    classifier_df = classifier_df[classifier_df['preprocessed_metric'] != '']

    return classifier_df

def gen_model(omni_conxn_str):
    '''
    - generates text classifying model based on data we currently have. classifies into cont, dist, unfunded, and nav. 
    - Currently combines end_nav and beg_nav to create a general 'nav' category. 
    - classifies by vectorizing strings, applying tfidf and then logistic regerssion 
    '''
    market_query = f"""
        SELECT
        *
        FROM
        aq_gp_cab_format
    """
    
    omni_conxn = create_engine(omni_conxn_str)
    omni_connection = omni_conxn.connect()
    
    cab_data_df = pd.read_sql(market_query, omni_connection)
    cab_data_df = cab_data_df[['beg_nav', 'contributions', 'distributions', 'end_nav', 'unfunded']]
    
    pivoted_df = pd.DataFrame()
    for col in cab_data_df.columns:
        #df = pd.DataFrame({'metric': list(cab_data_df[col]), 'label': [col for x in range(len(cab_data_df[col]))]})
        if col in ('end_nav', 'beg_nav'):
            df = pd.DataFrame({'metric': list(cab_data_df[col]), 'label': ['nav' for x in range(len(cab_data_df[col]))]})
        else: 
            df = pd.DataFrame({'metric': list(cab_data_df[col]), 'label': [col for x in range(len(cab_data_df[col]))]})
        pivoted_df = pd.concat([pivoted_df, df])
    
    pivoted_df = prediction_precleaning(pivoted_df, metric_col_name = 'metric')
    
    # splitting 
    X_train, X_test, y_train, y_test = train_test_split(pivoted_df['preprocessed_metric'], pivoted_df['label'], random_state=1087, test_size=0.2)
    
    lgclf = Pipeline([
                    ('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf', LogisticRegression(random_state=0)),
                  ])
    lgclf.fit(X_train, y_train)
    
    y_pred = lgclf.predict(X_test)    
    print('logistic classifier testing accuracy %s' % accuracy_score(y_pred, y_test))
    
    lgclf.fit(pivoted_df['preprocessed_metric'], pivoted_df['label'])
    return lgclf

def generate_pdf_img_dict(input_folder, file_list = None, inverted = False):
    pdf_to_image_dict = {}
    for file in os.listdir(input_folder):
        if file_list != None:
            if file not in set(file_list):
                continue
        #print(' - ' + file)
        #file_img = convert_from_path(os.path.join(input_folder, file), fmt='tiff')[0]
        doc  = fitz.open(os.path.join(input_folder, file))
        page = doc.load_page(0)
        file_img = page.get_pixmap(dpi=500)
        file_img = Image.frombytes("RGB", [file_img.width, file_img.height], file_img.samples)
        file_img = cv2.cvtColor(np.array(file_img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(file_img, cv2.COLOR_BGR2GRAY)
        _, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY) 
    
        # Morph open to remove noise and invert image
        if inverted:
            gray = 255 - gray
        pdf_to_image_dict[file] = gray

    return pdf_to_image_dict

def generate_text_data(pdf_image, min_confidence = 50, config = '--psm 4'):
    # https://stackoverflow.com/questions/61461520/does-anyone-knows-the-meaning-of-output-of-image-to-data-image-to-osd-methods-o
    data_df = pytesseract.image_to_data(pdf_image, output_type = pytesseract.Output.DATAFRAME, config = config)
    # limiting extractions to confidence level > 50%; consider increasing confidence level even more 
    data_df = data_df[data_df['conf'] >= min_confidence] 
    
    # combines the text with everything in an equal block/line/paragraph
    data_df['combined_block_text'] = data_df.groupby(['block_num','line_num', 'par_num'])['text'].transform(lambda x: ' '.join(x))
    data_df['group_index'] = data_df.groupby(['block_num','line_num', 'par_num']).ngroup()
    # creating coordinates column for ease of use with opencv later 
    data_df['coord'] = data_df.apply(lambda x: [x['left'], x['top'], x['width'], x['height']], axis = 1)
    # creating a list of coordinates of the items in the same block/line/paragraph
    df_combined_metrics = data_df.groupby(['block_num','line_num', 'par_num'])['coord'].apply(lambda x: list(x)).reset_index()
    data_df = pd.merge(
        left = data_df,
        right = df_combined_metrics,
        how = 'left', 
        on = ['block_num', 'line_num', 'par_num'],
        suffixes = ['_item', '_block_par_line_group']
    )
    
    data_df['final_coord_block_par_line_group'] = data_df['coord_block_par_line_group'].apply(lambda x: group_up_coord(x))

    return data_df

def get_predictions(pdf_image, data_df, model, min_confidence = .7):
    category_colors = {
        'unfunded': (255,0,0),
        'end_nav': (0,255,0),
        'distributions': (0,0,255),
        'contributions': (255, 255, 0),
        'beg_nav': (0, 255, 255),

        'nav': (255, 0 , 255)
    }

    cleaned_data_df = prediction_precleaning(data_df.drop_duplicates(['block_num', 'line_num', 'par_num']), metric_col_name = 'combined_block_text')
    
    row_df = cleaned_data_df.copy()[['preprocessed_metric', 'final_coord_block_par_line_group', 'group_index', 'combined_block_text']]

    print('getting category predictions...')
    row_df['predicted_category'] = model.predict(row_df['preprocessed_metric'].apply(lambda x: x.lower()))
    row_df['prediction_confidence'] = np.amax(model.predict_proba(row_df['preprocessed_metric'].apply(lambda x: x.lower())), axis = 1)
    selection_df = row_df.sort_values(['predicted_category', 'prediction_confidence'], ascending = False).groupby('predicted_category').head(5)
    print(' - categories found: ' + str(set(selection_df['predicted_category'])))

    high_confidence_df = selection_df[selection_df['prediction_confidence'] > min_confidence]
    best_guess_df =  selection_df.groupby('predicted_category').head(1)
    
    high_confidence_df_img = pdf_image.copy()
    high_confidence_df_img = cv2.cvtColor(high_confidence_df_img, cv2.COLOR_GRAY2RGB)

    for i in range(len(high_confidence_df)):
        coord = high_confidence_df['final_coord_block_par_line_group'].iloc[i]
        x, y, w, h = coord[0], coord[1], coord[2], coord[3]
        color = category_colors[high_confidence_df['predicted_category'].iloc[i]]
        cv2.rectangle(high_confidence_df_img, (x, y), (x + w, y + h), color, 5)
        cv2.putText(high_confidence_df_img, str(round(high_confidence_df['prediction_confidence'].iloc[i], 2)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
    
    img = pdf_image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    for i in range(len(best_guess_df)):
        coord = best_guess_df['final_coord_block_par_line_group'].iloc[i]
        x, y, w, h = coord[0], coord[1], coord[2], coord[3]
        color = category_colors[best_guess_df['predicted_category'].iloc[i]]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
        cv2.putText(img, str(round(best_guess_df['prediction_confidence'].iloc[i], 2)), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)
    
    return img, high_confidence_df_img, high_confidence_df, best_guess_df, row_df
  
def category_selection(x, threshold = .6):
    '''
    method used to make selections from prediction pool. Rules are as follows:
    - if the best prediction has a confidence > threshold, return all predictions w/ confidence > threshold 
    - else: if the category is a nav, return the top 2 predictions w/ highest confidence, else return the top prediction w/ highest confidence 
    '''
    sorted_df = x.sort_values(['prediction_confidence'], ascending = False)
    if sorted_df.iloc[0]['prediction_confidence'] >= threshold:  
        return sorted_df[sorted_df['prediction_confidence'] >= threshold]
    else:
        if sorted_df.iloc[0]['predicted_category'] == 'nav':
            return sorted_df.head(2) 
        else:
            return sorted_df.head(1)

def extract_row_data(raw_data_df, categorized_df):
    '''
    method that extracts numerical data in each row. It basically goes into the rawest data df, selects the data within valid categorized groups, 
    selects any data that's valid, and remerges it with valid_category groups 
    Steps:
        1. does general cleaning to the potential data (replaces 'SO', only keeps data that has at least 1 digit or '-', etc.
        2. remerges back to the categorized_df
    '''
    rows_df = raw_data_df[raw_data_df['group_index'].apply(lambda x: x in set(categorized_df['group_index']))]
    rows_df['text'] = rows_df['text'].apply(lambda x: '$0' if x == 'SO' else x) # hardcoded rule, might have to add/remove
    rows_df = rows_df[rows_df['text'].apply(lambda x: any(char.isdigit() for char in x) | (x == '-'))] # limits to rows with at least 1 number
    #rows_df = rows_df[rows_df['text'].apply(lambda x: is_date(x, fuzzy = True) == False)] # remove dates
    pulled_values_df = rows_df[['combined_block_text', 'text', 'group_index']]#.groupby('group_index').nth([-1])[['combined_block_text', 'text', 'group_index']] # assuming 1 value
    #pulled_values_df['text'] = pulled_values_df['text'].apply(lambda x: x.replace('S', '$').replace('s', '$').replace('O', '0').replace('o', '0'))

    extracted_row_df = pd.merge(
        left = categorized_df[['preprocessed_metric','group_index', 'predicted_category', 'prediction_confidence', 'final_coord_block_par_line_group', 'file']],
        right = pulled_values_df,
        on = 'group_index',
        how = 'right'
    )
    extracted_row_df = extracted_row_df.rename({'text': 'extracted_value'}, axis = 1)
    extracted_row_df = extracted_row_df.sort_values(['file','predicted_category', 'prediction_confidence'], ascending=[True, True, False])

    return extracted_row_df

def categorize_nav(df):
    # RECATEGORIZING NAVs AFTER NUMBERS HAVE BEEN PULLED; ASSUME FIRST TWO ROWS WITH NUMERIC VALUES ARE THE BEG_NAV/END_NAV
    # TODO: what happens if beg_nav is found and not end_nav or vice versa?
    def nav_categorizer(x, group_set): 
        if x['predicted_category'] != 'nav':
            return True
        else: 
            return x['group_index'] in group_set
    
    extracted_row_df = df.copy()

    # finds first two predicted rows (highest confidence)
    top_nav_groups_w_num = set(extracted_row_df[extracted_row_df['predicted_category'] == 'nav']['group_index'].drop_duplicates()[:2])
    extracted_row_df = extracted_row_df[extracted_row_df.apply(lambda x: nav_categorizer(x, top_nav_groups_w_num), axis =1)] # removing all navs except first two (top 2 confidence)

    # categorizes these 2 rows into beg_nav and end_nav based on x_coord
    nav_rows = extracted_row_df[extracted_row_df['predicted_category'] == 'nav']
    nav_rows['x_coord'] = nav_rows['final_coord_block_par_line_group'].apply(lambda x: x[1] if x == x else x)
    nav_rows = nav_rows.sort_values(['x_coord'])
    beg_nav_group_num = nav_rows[['group_index']].drop_duplicates()['group_index'].iloc[0]
    nav_rows['predicted_category'] = nav_rows['group_index'].apply(lambda x: 'beg_nav' if x == beg_nav_group_num else 'end_nav')
    extracted_row_df[extracted_row_df['predicted_category'] == 'nav'] = nav_rows

    return extracted_row_df

def extract_data(file, pdf_img_dict, classifier):
    '''
    a. pull text data using tesseract w/ function generate_text_data
    b. predict row categories using get_predictions()
    c. select rows based on predetermined rules defined in category_selection()
    d. narrow down rows even more by minimum threshold: currently defined as 40% 
    e. pull numerical data from each row 
    f. make selection from categorized nav rows into singular beg_nav and end_nav; this is done later b/c we only want to make selection from rows w/ numerical data
    g. generate a num_cols column which is basically the number of data points in each row. this is used to only select QTD data (only one column)
    h. predict whether QTD data is on the left or right side, narrow down data based on prediction. NOTE: this assumes QTD is only 1 column, if its 2+ columns you're SOL (itll always pick the first or last of each row)
    '''
    
    # 3a
    cleaned_data_df = generate_text_data(pdf_img_dict[file])

    #3b
    high_confidence_img, best_img, high_confidence_df, best_guess_df, raw_predictions_df = get_predictions(pdf_img_dict[file], cleaned_data_df, classifier)
    # ^ note that high_confidence_img, best_img, high_confidence_df, and best_guess_df are not currently necessary. potentially remove in the future 

    #3c
    categorized_df = raw_predictions_df.groupby('predicted_category').apply(lambda x: category_selection(x)).reset_index(drop = True).sort_values(['predicted_category', 'prediction_confidence'], ascending = [True, False])
    categorized_df['file'] = [file for x in range(len(categorized_df))]

    #3d
    prev_len = len(categorized_df)
    categorized_df = categorized_df[categorized_df['prediction_confidence'] > .4]
    if len(categorized_df) != prev_len:
        print(' - dropped some predicted values due to low confidence')
    categorized_df = categorized_df.sort_values(['predicted_category', 'prediction_confidence'], ascending = [True, False])

    #3e
    extracted_row_df = extract_row_data(cleaned_data_df, categorized_df)
    
    #3f
    extracted_row_df = categorize_nav(extracted_row_df)

    #3g
    extracted_row_df = extracted_row_df.drop(['final_coord_block_par_line_group'], axis = 1)
    extracted_row_df = extracted_row_df.sort_values(['file','predicted_category', 'prediction_confidence'], ascending=[True, True, False])
    extracted_row_df = pd.merge(
        left = extracted_row_df, 
        right = extracted_row_df.groupby('group_index').size().rename('num_cols'), 
        on = 'group_index'
    )

    #3h 
    # guessing quarter location (left or right)
    location_guess = generalize_quarter_area(cleaned_data_df)
    print(' - data we want is on the \'' + location_guess + '\' side')
    if 'left' in location_guess:
        narrowed_df = extracted_row_df.drop_duplicates('group_index', keep = 'first')
    else:
        narrowed_df = extracted_row_df.drop_duplicates('group_index', keep = 'last')

    extracted_row_df['guessed_quarter_loc'] = [location_guess for _ in range(len(extracted_row_df))]

    return cleaned_data_df, raw_predictions_df, categorized_df, extracted_row_df, narrowed_df

# helper method for generalize_quarter_area() specifically
def compare_y_coord(x):
        if x['quarter_coord'][0] < x['year_coord'][0]:
            return 'left'
        else:
            return 'right'
            
def generalize_quarter_area(cleaned_data_df):
    potential_quarter_data = cleaned_data_df[cleaned_data_df['text'].apply(lambda x: any(y in x.lower() for y in ['period','current','quarter', 'qtd']))]
    potential_year_data = cleaned_data_df[cleaned_data_df['text'].apply(lambda x: any(y in x.lower() for y in ['year', 'ytd', 'itd']))]
    
    potential_groups = set(potential_quarter_data['group_index']) & set(potential_year_data['group_index'])
    potential_quarter_data = potential_quarter_data[potential_quarter_data['group_index'].apply(lambda x: x in potential_groups)].drop_duplicates('combined_block_text')
    potential_year_data = potential_year_data[potential_year_data['group_index'].apply(lambda x: x in potential_groups)].drop_duplicates('combined_block_text')
    #print(' - number of quarter options: ' + str(len(potential_quarter_data)))
    #print(' - number of year options: ' + str(len(potential_year_data)))
    guessed_answer = 'left*'
    
    if len(potential_quarter_data) != 0 and len(potential_year_data) != 0:
        decision_df1 = potential_quarter_data[['group_index','coord_item']].rename({'coord_item':'quarter_coord'}, axis = 1)
        decision_df2 = potential_year_data[['group_index','coord_item']].rename({'coord_item':'year_coord'}, axis = 1)
        merged_decision_df = pd.merge(left = decision_df1, right = decision_df2, on = 'group_index')
        merged_decision_df['decision'] = merged_decision_df.apply(lambda x: compare_y_coord(x), axis = 1)
        #print(' - ' + str(set(merged_decision_df['decision'])))
        if len(set(merged_decision_df['decision'])) == 1:
            guessed_answer = merged_decision_df['decision'][0]
        else:
            guessed_answer = 'left**'

    return guessed_answer

# debugging method
def debug_visual(pdf_image, coord, label = None, color = (255, 255, 255)):
    img = pdf_image.copy()
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    x, y, w, h = coord[0], coord[1], coord[2], coord[3]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)
    if label:
        cv2.putText(img, str(label), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,0), 2)

    return img

# helper method 
def group_up_coord(x):
    left = x[0][0]
    top = x[0][1]
    width = int(x[-1][0] + x[-1][2] - x[0][0]) #int(sum([x[iter][2] for iter in range(len(x))])) #<- cant use this: it doesnt account for white space in between
    height =  max([x[iter][3] for iter in range(len(x))]) #int(sum([x[iter][3] for iter in range(len(x))])/len(x)) # <- average, could potentially be better

    return [left, top, width, height]

# helper method 
def is_date(string, fuzzy=False):
    """
    helper function; Return whether the string can be interpreted as a date.

    :param string: str, string to check for date
    :param fuzzy: bool, ignore unknown tokens in string if True
    """
    try: 
        parse(string, fuzzy=fuzzy)
        return True

    except ValueError:
        return False
 
def main():
    '''
    1. generate text training model 
    2. generate pdf_img_dict (which converts all PDFs into images and adds them into a PDF:img dict)
    3. iterate through each file and extract data

    TODO: should we apply 3f to all categories? AKA should we move step 3c AFTER 3f?

    '''
    omni_conxn_str = 'mysql+pymysql://bwong:?(}9LcsW@analytics-proxy.stepstoneapps.com/spar_analytics'
    lgclf = gen_model(omni_conxn_str)
    print()

    file_list = None
    if True: # example of sub-sample of files
        ishaan_df = pd.read_excel('cab_validation_ishaan.xlsx')
        ishaan_df = ishaan_df[ishaan_df['Assigned'] == 'Ishaan']
        file_list = list(set(ishaan_df['file']))
        file_list.sort()
    input_folder = 'text_extracting_folder_test'
    pdf_img_dict = generate_pdf_img_dict(input_folder, file_list = file_list)

    narrowed_final_df = pd.DataFrame()
    expanded_final_df = pd.DataFrame()
    categorized_df_dict = {} # used for debugging 
    for file in file_list:
        print(file)
        categorized_df, narrowed_df, extracted_row_df = extract_data(file, pdf_img_dict, lgclf)

        categorized_df_dict[file] = categorized_df # used for debugging; has raw data before doing column-data extraction 
        narrowed_final_df = pd.concat([narrowed_final_df, narrowed_df])
        expanded_final_df = pd.concat([expanded_final_df, extracted_row_df])
        #break
    
    narrowed_final_df.to_excel('results_narrow_df.xlsx')
    expanded_final_df.to_excel('results_expanded_df.xlsx')

if __name__ == "__main__":
    main()