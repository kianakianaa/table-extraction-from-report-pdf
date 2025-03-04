import camelot
import pandas as pd
import re
import numpy as np
import pdfplumber
import matplotlib.pyplot as plt
from collections import Counter
from pathlib import Path


def detect_extract_tables(pdf_path, save_tables=True):
    print('Begin reading pdf...\n')
    
    table_pages = detect_tables(pdf_path)
    print('Finish detecting tables!')
    # table_pages = [45]  # combine first few rows 
    
    table_num = 0
    df_dict_list_all = []
    for page_num in table_pages:
        print(f'\nProcessing page {page_num}..')
        df_dict_list_page = get_page_tables_adjusted(pdf_path, page_num, save_tables=False, start_idx=table_num)
        table_num += len(df_dict_list_page)
        df_dict_list_all += df_dict_list_page
    
    if save_tables:
        xlsx_name = Path(pdf_path).stem + '_all' + '.xlsx'
        xlsx_dir = './output/table'
        xlsx_path = Path(xlsx_dir) / xlsx_name
        xlsx_path.parent.mkdir(parents=True, exist_ok=True) 
        save_tables_to_xlsx(df_dict_list_all, xlsx_path, start_idx=0)
    
    return df_dict_list_all  


def get_page_tables_adjusted(pdf_path, page_num, save_tables=True, start_idx=0):
    # extract: tablelist has no title or columns detected, pure list of df
    table_dict, df_dict_flavors = get_best_table_camelot(pdf_path, ",".join(map(str, [page_num+1])))  
    
    # refine/cleaning: go through each one
    # element in df_dict_list = {'title': title, 'df': df, 'page': page_string, bbox':bbox} for each table
    df_dict_list = [clean_table_df(table_df) for table_df in table_dict['df']] 
    for i in range(len(table_dict['df'])):
        df_dict_list[i]['bbox'] = table_dict['bbox'][i]
        df_dict_list[i]['page'] = page_num
    df_dict_list = [df_dict for df_dict in df_dict_list if not df_dict['df'].empty]
    
    if len(df_dict_list) == 0:
        print(f'No table for page {page_num}')
        return []
    else:
        print(f'num of cleaned df: {len(df_dict_list)}')
    
    # update bbox
    # refined_plumber_bbox = find_table_position(table_df, pdf_file, table_bbox, page_num=0)
    df_dict_list_1 = list(map(lambda df_dict: {**df_dict, 'bbox': find_table_position(df_dict['df'], pdf_path, df_dict['bbox'], page_num)}, df_dict_list))
    # adjust column order  (page_num=0)
    df_dict_list_2 = list(map(lambda df_dict_1: {**df_dict_1, 'df': adjust_column_name_order(df_dict_1['bbox'], df_dict_1['df'], pdf_path, page_num)}, df_dict_list_1))
    if [df_dict for df_dict in df_dict_list_2 if len(df_dict['df'])>0]:
        print(f'page {page_num} got final df!')
        
    
    #  save df_dict_list into xlsx, each df_dict into an independent sheet, with the sheetname being its title (if title is none then just use Sheet + number)
    if save_tables:
        xlsx_name = Path(pdf_path).stem + '.xlsx'
        xlsx_dir = './output/table'
        xlsx_path = Path(xlsx_dir) / xlsx_name
        xlsx_path.parent.mkdir(parents=True, exist_ok=True) 
        save_tables_to_xlsx(df_dict_list_2, xlsx_path, start_idx)
    
    return df_dict_list_2


def detect_tables(pdf_path):
    table_pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            if tables:
                dfs = [pd.DataFrame(table) for table in tables if any(any(cell for cell in row) for row in table)]
                if dfs:
                    table_pages.append(i)
    return table_pages


def read_tables_camelot(src, flavor, pages):
    tables = camelot.read_pdf(src, flavor=flavor, pages=pages, suppress_stdout=True)
    table_list = [] 
    position_list = []  
    
    for table in tables:
        table_df = table.df
        table_list.append(table_df)
        position_list.append(table._bbox)  # (x1, y1, x2, y2)
    
    return table_list, position_list

def get_best_table_camelot(pdf_path, pages='all'):
    
    def separate_tables(df):
        first_text = df.iloc[0].dropna().tolist()
        first_text = [x for x in first_text if x != '']
        if len(first_text)>=4:
            if len(first_text)%2 == 0:
                first_half = first_text[:int(len(first_text)/2)]
                second_half = first_text[int(len(first_text)/2):]
                if len(first_half) >2 and first_half[1:] == second_half[1:]:
                    element = str(first_half[-1])
                    match = df.iloc[0].apply(lambda x: str(x).strip() == element)
                    true_indices = match[match].index
                    if true_indices.any():
                        # print(f'match:{true_indices}')
                        column_index = true_indices[0]
                        # print(f'column_index:{column_index}')
                        df_list = [df.iloc[:, :column_index+1], df.iloc[:, column_index+1:]]
                        # print(f'separate:\n{df_list[0]}')
                        return df_list 
            else:
                first_half = first_text[:int((len(first_text)-1)/2)]
                second_half = first_text[int((len(first_text)-1)/2):]
                if len(set(first_half) & set(second_half)) == (len(first_text)-1)/2:
                    match = df.iloc[0].apply(lambda x: str(x).strip() == element)
                    true_indices = match[match].index
                    if true_indices.any():
                        # print(f'match:{true_indices}')
                        column_index = true_indices[0]
                        df_list = [df.iloc[:, :column_index+1], df.iloc[:, column_index+1:]]
                        # print(f'\nseparate tables:\n{df_list[1]},\n{df.iloc[1]}')
                        return df_list
        
        # although there is no repeatec column names, repeated None structure
        # Iterate over each row to find the column where the NaN starts and all after are NaN
        def find_column_index(row):
            if len(row) <=4:
                return None
            for i in range(1, len(row)-1):
                if pd.isna(row[i]):  # If the value is NaN
                    # Check if all values after this column are also NaN
                    if all(pd.isna(row[i:])):
                        if i <=3:
                            return None
                        else:
                            return i
            return None
        num = df.shape[0]
        df_1 = df.replace({None: pd.NA, '': pd.NA})
        df_1.columns = list(range(df.shape[1]))
        susp_col_idx = df_1.apply(find_column_index, axis=1)
        susp_col_idx = [x for x in susp_col_idx if pd.notna(x) and str(int(x)).isdigit()]
        # print(f'susp_col_idx:{susp_col_idx}')
        if len(set(susp_col_idx))<=5 and len(susp_col_idx)/num>0.3 and len(susp_col_idx)>=10:
            col = int(sorted(susp_col_idx)[0])
            print(f'Find start of NAN col:{col}')
            df_list = [df.iloc[:, :col], df.iloc[:, col:]]
            return df_list
        
        # check the last column
        if df.shape[1] > 4:
            df_last = df.iloc[1:, -1].replace({None: pd.NA, "": pd.NA}).dropna()
            if len(df_last) < 3:
                return [df]
            # print(f'df_last: \n{df_last}')
            df_count_n = df_last.apply(lambda x: x.count('\n'))
            num = len([e for e in df_count_n.values if e>0])
            perc = num/len(df_last)
            # print(f'df_count_n:\n{df_count_n}', '\n', perc)
            if len(set(df_count_n.values)) <= 2 and perc>=0.8: 
                print('Separate last column!')
                df_1 = df.iloc[:, :-1]
                df_2 = df.iloc[:, -1]
                df_split = df_2.str.split('\n', expand=True)
                df_split.columns = [i for i in range(df_split.shape[1])]
                df_list = [df_1, df_split]
                return df_list
            
                
        return [df]
    
    def cal_none_perc(df):
        num_none = df.isna().sum().sum()
        num_total = df.shape[0] * df.shape[1]
        none_perc = num_none/num_total
        return none_perc
    
    def examine_table(df, string_thre = 0.5, string_none_thre = 0.15):
        """ 
        Goal: remove invalid values 
        if the whole table is invalid, will return an empty dataframe.
        """
        
        def check_first_col(df):
            df_col = df.iloc[:, 0]
            none_val = df_col.isna().sum()
            start_with_num = df_col.str.match(r'^\d').sum() - df_col.str.match(r'^\d{4}$').sum()
            denominator = len(df_col) - none_val
            if denominator == 0:
                denominator = 0.001
            invalid_perc = (start_with_num)/denominator
            if invalid_perc > 0.5:
                return pd.DataFrame()
            else:
                return df
        
        def check_strings(df): 
            # helper function
            def check_punctuation(df_obj):
                df_obj = df_obj.apply(lambda col: col.map(lambda x: x.rstrip('.') if isinstance(x, str) else x))
                invalid_punc_df = pd.DataFrame(0, index=df_obj.index, columns=df_obj.columns)
                def has_non_float_dot(value):
                    if not isinstance(value, str) or '.' not in value:
                        return 0  # Ignore non-strings or values without a dot
                    float_pattern = re.compile(r'^\d+\.\d+$')  
                    if float_pattern.fullmatch(value):  
                        return 0  # Directly return 0 if it's a float
                    # Check if at least one `.` is NOT part of a float
                    for i, char in enumerate(value):
                        if char == '.':
                            # If it's the first or last character, it's clearly not a decimal point
                            if i == 0 or i == len(value) - 1:
                                return value
                            # If the character before OR after is not a digit, it's not a decimal point
                            if not (value[i-1].isdigit() and value[i+1].isdigit()):
                                return value  # Found a non-float dot, return the value
                    return 0
                    
                for col in df_obj.columns:
                    df_col = df_obj[col]
                    values_array = df_col.values
                    index_list = df_col.index
                    punc_list = [has_non_float_dot(v) for v in values_array] 
                    if punc_list:
                        invalid_list = [index_list[i] for i in range(len(punc_list)) if punc_list[i]!=0]
                        invalid_punc_df.loc[invalid_list, col] = 1
                return invalid_punc_df  # invalid_perc_df 
            
            def check_capital_pattern(df_obj):
                df_invalid = pd.DataFrame(0, index=df_obj.index, columns=df_obj.columns)
                df_cap = df_obj.map(lambda x: int(x[0].isupper()) if isinstance(x, str) and x and not x[0].isdigit() else x) # only check string upper/lower
                df_count_pattern = pd.DataFrame([[0]*df_cap.shape[1]], columns=df_cap.columns)
                for col in df_cap.columns:
                    df_cap_col = df_cap[col]
                    # remove irrelevant part
                    df_cap_col = df_cap_col[df_cap_col != -1] 
                    indices = df_cap_col.index.to_list()
                    values = df_cap_col.tolist()
                    # counting
                    valid_count = 0
                    in_sequence = False
                    seq_start_idx = None  
                    temp_count = 0  # to remove the isolated 1
                    seq_indices = []
                    
                    for i in range(len(values)): # iterate over the list
                        idx = indices[i]
                        if df_cap_col[idx] == 1:
                            seq_indices = [idx]
                            in_sequence = True
                            seq_start_idx = idx
                            temp_count = 1  # Start counting sequence

                        elif df_cap_col[idx] == 0:
                            if in_sequence:  # Continue valid sequence
                                seq_indices.append(idx)
                                temp_count += 1
                            # Ensure index continuity
                            if seq_start_idx is not None and idx > seq_start_idx:
                                if indices[i] != indices[i - 1] + 1:  # Non-contiguous index
                                    in_sequence = False  # Invalidate sequence
                                    temp_count = 0
                                    seq_indices = []
                                    
                        else:
                            if in_sequence and df_cap_col[indices[i-1]]==0:
                                temp_count += 1
                        if in_sequence and (i == len(values) - 1 or values[i + 1] == 1):
                            if temp_count > 2:  # Ensure at least two 0 exists
                                valid_count += temp_count  # number of values that follow the pattern
                                df_invalid.loc[seq_indices,col] = 1
                            in_sequence = False
                            temp_count = 0
                            
                    df_count_pattern[col] = valid_count
                
                return df_invalid # df_count_pattern
            
            
            def is_obj_column_valid(df, col):
                # Check if the column is of dtype object and contains less than 10% numeric values or digit strings
                df_col = df[col]
                df_col = df_col.str.replace(r'\s+', ' ', regex=True).str.strip()
                df_col = df_col.str.replace('\xa0', '', regex=True)
                df_col = df_col.astype(str).apply(lambda x: x.replace('\n', '').replace('-', '').strip())
                digit_str_count = df_col.str.match(r'^-?[\d,]+(\.\d+)?$').sum()
                none_value = df_col.isna().sum()
                total_count = len(df_col) - none_value
                if total_count == 0:
                    total_count = 0.01
                # Calculate the percentage of digit strings
                digit_percentage = (digit_str_count / total_count) * 100

                return digit_percentage < 10
            
            obj_cols = df.select_dtypes(include=['object']).columns
            obj_cols = [col for col in obj_cols if is_obj_column_valid(df, col)]
            # print(f'obj_col:{obj_cols}')
            df_obj = df[obj_cols].map(lambda x: re.sub(r'^\W+', '', x) if isinstance(x, str) else x)
            num_entries = df.shape[0]
            none_array = df_obj.isna().sum().values.reshape(1, -1)
            # check capital pattern
            df_pattern_invalid = check_capital_pattern(df_obj)
            # check punctuation
            df_punc_invalid = check_punctuation(df_obj)
            # combine and calculate perc
            df_combined_invalid = df_pattern_invalid + df_punc_invalid
            df_combined_invalid = df_combined_invalid.replace({1: None, 2: None})
            invalid_array = df_combined_invalid.isna().sum().values.reshape(1, -1)
            epsilon = 1e-10  # Small value to avoid division by zero
            denominator = num_entries - none_array
            denominator = np.where(denominator == 0, epsilon, denominator)
            invalid_str_perc_array = (invalid_array)/denominator
            df_invalid_obj = pd.DataFrame(invalid_str_perc_array, columns=obj_cols)
            invalid_perc_array =  (invalid_array)/(num_entries)
            df_invalid_all = pd.DataFrame(invalid_perc_array, columns=obj_cols)
            
            return df_invalid_obj, df_invalid_all
        

        # missing values
        df = df.replace(r'^\s*$', None, regex=True)
        if df.empty:
            return df
        else:
            df_check_col = check_first_col(df)
            if df_check_col.empty:
                return df_check_col
            
            # check string: capital letter and punctuation
            df_invalid_obj, df_invalid_all = check_strings(df)
            # Q: take the idea of forest (voting) for the cols?
            for col in df_invalid_obj.columns[1:]:
                if df_invalid_obj[col][0] >= string_thre or df_invalid_all[col][0] >= string_none_thre:
                    if df[col][:3].str.lower().isin(['unit']).any():
                        continue
                    # print(df_invalid_obj[col][0], df_invalid_all[col][0])
                    # print(df[col])
                    df = df.drop(columns=[col])
                    
                else:
                    # check none_perc 
                    none_perc = df[col].isna().sum()/df.shape[0]
                    if  none_perc >= 0.8:
                        df = df.drop(columns=[col])
            
            # avoid df having too many strings
            lengths = df.map(lambda x: len(str(x)) if pd.notna(x) else 0)
            avg_len = lengths.sum().sum()/(df.map(lambda x: 1 if pd.notna(x) else 0).sum().sum())
            # print(f'avg_len:{avg_len}')
            if avg_len >= 40:
                df = pd.DataFrame()
            if not df.empty and df.iloc[-1, 0] and df.iloc[-1, 0].count('\n')>3:
                df = pd.DataFrame()
            
        return df
    
    flavor_list = ['hybrid', 'lattice']  # , 'stream', 'network'
    flavor = 'hybrid' # by default
    
    # get results for different flavor
    df_dict = {key: [] for key in flavor_list}
    for flavor in flavor_list:
        # print(f'\nfit flavor {flavor}')
        df_list_0, bbox_list_0 = read_tables_camelot(pdf_path, flavor, pages)
        df_valid_list = []
        bbox_valid_list = []
        # separate tables
        df_list = []
        bbox_list = []
        for i, df in enumerate(df_list_0):
            separated_table_list = separate_tables(df)
            df_list += separated_table_list
            if len(separated_table_list) == 1:
                bbox_list.append(bbox_list_0[i])
            else:
                bbox_list += [bbox_list_0[i]]*2
            
        for i, df in enumerate(df_list):
            # check validness of df
            df = examine_table(df)
            df = df.dropna(how='all')
            df = df.dropna(axis=1, how='all')
            if not df.empty and df.shape[1]>=2 and df.shape[0]>=2:
                df_valid_list.append(df)
                bbox_valid_list.append(bbox_list[i])
        df_dict[flavor] = {'df':df_valid_list, 'bbox': bbox_valid_list, 'page': pages}
        
    # drop out if no df identified
    non_empty_flavor = [key for key in df_dict if len(df_dict.get(key).get('df'))>0]
    if len(non_empty_flavor)==1:
        flavor = non_empty_flavor[0]
    else:
        # check the missing value percentage
        # step 1: compare general missing percentage 
        none_perc_dict = {key: [cal_none_perc(df) for df in df_dict[key]['df']] for key in non_empty_flavor}
        overall_none_perc_dict = {key: np.mean(none_perc_dict[key]) for key in non_empty_flavor}
        min_none_perc = {key: value for key, value in overall_none_perc_dict.items() if value == min(overall_none_perc_dict.values())}
        if len(min_none_perc) == 1:
            flavor = list(min_none_perc.keys())[0]
        # step 2: still cannot decide then check the variance (distribution)
        else:
            var_none_perc_dict = {key: np.var(none_perc_dict[key]) for key in non_empty_flavor}
            min_var = {key: value for key, value in var_none_perc_dict.items() if value == min(var_none_perc_dict.values())}
            if len(min_var) == 1:
                flavor = list(min_var.keys())[0]
                
    print(f'Flavor chosen: {flavor}')      
    final_table_dict = df_dict[flavor]
    
    return final_table_dict, df_dict # df_dict

def is_non_year_numeric(value):
    # Remove commas (thousands separators)
    value = str(value).replace(',', '')
    # Try to convert the value to a float
    try:
        # Convert to float
        num_value = float(value)
        if len(str(int(num_value))) == 4:  # check if the number has 4 digits, likely a year
            return False
        return True
    except ValueError:
        # If it can't be converted to a float, return False
        return False


# cleaning
def clean_table_df(df):
    # identify title (maybe include the title of the page)
    def identify_title_columns(df):
        if df.shape[0]<3:
            return pd.DataFrame(), '', ''
        title = None
        cols_text = ''
        num_cols = df.shape[1]
        first_row = df.iloc[0].fillna('').tolist()
        second_row = df.iloc[1].fillna('').tolist()
        third_row = df.iloc[2].fillna('').tolist()
        combined_2_row = [[first_row[i], second_row[i]] for i in range(num_cols)]
        combined_2_row_c = [(0 if l.count('')==1 else 1) for l in combined_2_row]
        
        first_row_text = '_'.join(first_row)
        first_row_nume = sum([int(is_non_year_numeric(e)) for e in first_row])
        num_lines_1 = first_row_text.count('\n')
        second_row_text = '_'.join(second_row)
        second_row_nume = sum([int(is_non_year_numeric(e)) for e in second_row])
        num_lines_2 = second_row_text.count('\n')  
        third_row_text = '_'.join(third_row)   
        third_row_nume = sum([int(is_non_year_numeric(e)) for e in third_row]) 
        num_lines_3 = third_row_text.count('\n') 
        # Case 2: both first and second row has None, but can be combined into one row -> no title, first two rows merge into cols
        if sum(combined_2_row_c) == 0:
            print('First two rows as cols')
            cols = [str(l[0]) + str(l[1]) for l in combined_2_row]
            df_1 = df.iloc[2:]
            df_1.columns = cols
        # Case 1: no missing values in first row -> column names, no title
        elif first_row[1:].count('') == 0 and 2* num_cols > num_lines_1 >=0 and first_row_nume<=2:
            print('First row as cols!')
            df_1 = df.iloc[1:]
            cols = first_row
            df_1.columns = cols
        
        # Case 3: two many \n in the first row: first row as cols combined, no title
        elif 2*num_cols > num_lines_1 > max(num_cols-2, 2):
            print('First row as cols, tangled!')
            cols_text = first_row_text.strip('_')
            df_1 = df.iloc[1:]  # df_1.columns = indexRange
        # Case 4: first row not cols, see if the second is cols  (\n count)
        elif 2*num_cols > num_lines_2 >=0 and second_row[1:].count('')==0 and second_row_nume<=2: 
             # first two rows are not titles nor cols, third row has no missing values (assume to be col names)
            print('Second row as cols!')
            title = first_row_text.strip('_')
            cols = second_row
            df_1 = df.iloc[2:]
            df_1.columns = cols
        elif 2*num_cols > num_lines_1 > max(num_cols-2, 2):
            print('Second row as cols, tangled!')
            # print(num_lines_2)
            cols_text = second_row_text.strip('_')
            title = first_row_text.strip('_')
            df_1 = df.iloc[2:]
       
        elif third_row[1:].count('')==0 and  2*num_cols> num_lines_3 >=0 and third_row_nume<=2:
            print('Thrid row as cols')
            title = "|".join([str(l[0]) + '_' + str(l[1]) for l in combined_2_row]).strip('_').strip('|')
            cols = df.iloc[2].tolist()
            df_1 = df.iloc[3:]
            df_1.columns = cols
        else:
        # by default: no change (no column names in first three rows)   
            df_1 = df.copy()
            # df_1.columns = list(range(df_1.shape[1]))
        
        return  df_1, title, cols_text
    
    # identify column names: when cols_text is not ''
    def format_columns(df_1, cols_text):
        df_2 = df_1.copy()
        # identify how many name are multi-rows (assume at most has two lines)
        col_list = re.split(r'\n+', cols_text)
        col_list = [e.strip() for e in col_list if not all(ch == '_' for ch in e.strip())]
        num = len(col_list)
        num_cols = df_1.shape[1]
        num_multi_lines = num - num_cols
        
        if num_multi_lines > 0:
            single_row_cols = col_list[num_multi_lines:num-num_multi_lines]
            start = col_list[:num_multi_lines]
            end = col_list[-num_multi_lines:]
            multi_line_cols = [start[i] + '_' + end[i] for i in range(num_multi_lines)]
            # problem: no adjust for order (for now just assume multile line is at the end) -> entails some ML to match
            new_cols = single_row_cols + multi_line_cols
        elif num_multi_lines == 0:
            new_cols = col_list
        else:
            # missing cols
            new_cols = col_list + list(range(num_multi_lines))
            # Q: just leave it empty?
            
        df_2.columns = new_cols[:num_cols]
        
        return df_2 
    
    # combined rows ('\n): by detecting the combination of None with neighbors
    def combine_rows(df_2):
        
        def detect_row(df, row):
            def is_punctuation(char):
                # Regex pattern: match if the character is not alphanumeric
                return bool(re.match(r'[^a-zA-Z0-9]', char))
            def is_only_punctuation(row):
                return all(is_punctuation(str(cell)) for cell in row)
            
            non_cols  = df.iloc[row].isna().tolist()
            check_cols = [idx for idx in range(1, df.shape[1])]  # if non_cols[idx]==0]
            df_up = df.iloc[row - 1].fillna('_').tolist()
            df_row = df.iloc[row].fillna('_').tolist()
            df_dw = df.iloc[row+1].fillna('_').tolist()
            num = df.shape[1]
            if all(df_up[idx] == '_' for idx in check_cols) and all(df_dw[idx] == '_' for idx in check_cols):
                df_bf = df.iloc[:row-1]
                df_af = df.iloc[row+2:]
                df_md = pd.DataFrame([[df_up[i] + df_row[i] + df_dw[i] for i in range(num)]], columns=df.columns)
                df_md = df_md.apply(lambda col: col.str.replace('_', ' '))
                df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
            elif all(df_up[idx] == '_' for idx in check_cols):
                df_bf = df.iloc[:row-1]
                df_af = df.iloc[row+1:]
                df_md = pd.DataFrame([[df_up[i] + df_row[i] for i in range(num)]], columns=df.columns)
                df_md = df_md.apply(lambda col: col.str.replace('_', ' '))
                df_new = pd.concat([df_bf, df_md, df_af], axis=0, ignore_index=True).reset_index(drop=True)
            # add: the whole row only has punctuation but no real stuff
            elif all(is_only_punctuation(str(cell)) for cell in df_row):
                df_bf = df.iloc[:row]
                df_af = df.iloc[row+1:]
                df_new = pd.concat([df_bf, df_af], axis=0, ignore_index=True)   
            else:
                df_new = df.copy()
            
            df_new = df_new.reset_index(drop=True)
            
            return df_new

        df_3 = df_2.copy()
        row = 0
        while row < df_3.shape[0]-1:
            df_3 = detect_row(df_3, row)
            row += 1
                
        return df_3
    
    # identify (segament)
    df_1, title, cols_text = identify_title_columns(df)
    # format columns: maybe wrong order (can include the column name in the later ML for matching)
    if df_1.empty:
        return {'title': '', 'df': df_1}
    if cols_text!='':
        df_2 = format_columns(df_1, cols_text)
    else:
        df_2 = df_1.copy()
    # identify and combine rows
    df_3 = combine_rows(df_2)
    df_dict = {'title': title, 'df': df_3}
    
    return df_dict


def find_table_position(table_df, pdf_file, table_bbox, page_num=0):
    """
    Find the bounding box of a table by identifying the first and last rows 
    within a given bounding box from Camelot, converting coordinates for pdfplumber.

    Args:
    - table_df (pd.DataFrame): The cleaned table (input as a DataFrame).
    - pdf_file (str): Path to the PDF file.
    - table_bbox (tuple): The list of bounding boxes from Camelot (x0, y0, x1, y1).
    - page_num (int): The page number where the table is located (default is 0).

    Returns:
    - tuple: pdfplumber ordinated bbox (top-left corner, bottom-right corner) of the refined table.
    """
    # If table_bbox is a tuple, we can choose the first bounding box
    if table_bbox:
        x0_table, y0_table, x1_table, y1_table = table_bbox
    else:
        print('Empty bbox')
        return None  # Return None if table_bbox is empty

    if table_df.empty:
        return None
    else:
        if table_df.shape[0]< 2 or table_df.shape[1]<2:
            return None
    
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[page_num]  
        # display_table_box(pdf_path, table_bbox, page_num=page_num, plumber_ordinate=0)
        # Convert Camelot bbox to pdfplumber coordinate system
        page_height = page.height  
        # print(page_height)
        y0_table_plumber = page_height - y1_table
        y1_table_plumber = page_height - y0_table

        # Extract words only within the transformed bbox
        words = [
            word for word in page.extract_words()
            if x0_table <= word['x0'] <= x1_table and y0_table_plumber <= word['top'] <= y1_table_plumber
        ]
        # display_table_box(pdf_file, (x0_table, y0_table_plumber, x1_table, y1_table_plumber), page_num=page_num, plumber_ordinate=1)
        if not words:
            print(f'empty words')
            return None
        
        def get_bbox_from_mode_or_m(bboxes, axis):
            """Calculate the mode of bounding box coordinates for robustness,
            and fall back to the median if no repetition is found.
            """
            if axis == 0:
                a0_col = 'top'
                a1_col = 'bottom'
            elif axis == 1:
                a0_col = 'x0'
                a1_col = 'x1'
                
            a0_vals = [round(bbox[a0_col], 2) for bbox in bboxes]
            a1_vals = [round(bbox[a1_col], 2) for bbox in bboxes]
            
            def get_mode_or_max(vals):
                mode = Counter(vals).most_common(1)
                if mode and mode[0][1] / len(vals) > 0.4:
                    return mode[0][0]
                return max(vals)
            
            def get_mode_or_min(vals):
                mode = Counter(vals).most_common(1)
                if mode and mode[0][1] / len(vals) > 0.4: 
                    return mode[0][0]
                return min(vals)
            
            a0_result = get_mode_or_min(a0_vals)
            a1_result = get_mode_or_max(a1_vals)
            # print(f'a0, a1: {a0_result, a1_result}')

            return (a0_result, a1_result)
        
        
        def locate_axis_boundary(table_df, axis, words):
            """
            axis = 0: row
            axis = 1: column
            """
            if axis==0:
                first_text = table_df.columns.dropna().tolist()
                first_idx = -1
                if all(isinstance(x, int) and x < 100 for x in first_text) or len(table_df.columns.dropna().tolist())<2:
                    first_idx = None
                    for i in range(table_df.shape[0]):  # Iterate over all columns
                        if len(table_df.iloc[i].dropna().tolist()) > 0:
                            first_idx = i
                            break
                    
                if first_idx is None: 
                    return None, None   
                elif first_idx != -1:
                    first_text = table_df.iloc[first_idx].dropna().tolist()
                
                last_idx = None
                for i in range(1, table_df.shape[0] + 1):  # Iterate from the last column
                    if len(table_df.iloc[-i].dropna().tolist()) >= 2:
                        last_idx = -i
                        last_text = table_df.iloc[last_idx].dropna().tolist()
                        break

                if last_idx is None:
                    return None, None   
                
            elif axis==1:
                # print(table_df.shape, table_df.iloc[:2])
                first_idx = None
                for i in range(table_df.shape[1]):  # Iterate over all columns
                    if len(table_df.iloc[:, i].dropna().tolist()) > 0:
                        first_idx = i
                        first_text = table_df.iloc[:, first_idx].dropna().tolist()
                        break
                if first_idx is None:  # If no non-empty column is found
                    return None, None   
                
                last_idx = None
                for i in range(1, table_df.shape[1] + 1):  # Iterate from the last column
                    if len(table_df.iloc[:, -i].dropna().tolist()) >= 2:
                        last_idx = -i
                        # print(f'found last_idx:{last_idx}')
                        last_text = table_df.iloc[:, last_idx].dropna().tolist()
                        # print(f'last_text:{last_text}')
                        break
                if last_idx is None:
                    return None, None   
                

            else:
                print('Wrong axis!')
                return None, None   
            
            first_text = [str(e).split("\n")[0] for e in first_text]
            first_text = [word for e in first_text for word in re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\w+\b', str(e))] # prevent separate float
            first_text = [e for e in first_text if len(e)>1]
            last_text = [str(e).split("\n")[-1] for e in last_text]  # deal with \n
            last_text = [word for e in last_text for word in re.findall(r'\b\d{1,3}(?:,\d{3})*(?:\.\d+)?|\w+\b', str(e))] # re.findall(r'\b\w+\.\w+|\w+\b'
            last_text = [e for e in last_text if len(e)>1]            
            first_bboxes, last_bboxes = [], []
            for word in words:
                if any(str(cell) in word['text'] for cell in first_text):
                    first_bboxes.append(word)
                if any(str(cell) in word['text'] for cell in last_text):
                    last_bboxes.append(word)
            if not first_bboxes or not last_bboxes:
                return None, None           
            
            first_bbox_mode = get_bbox_from_mode_or_m(first_bboxes, axis)
            last_bbox_mode = get_bbox_from_mode_or_m(last_bboxes, axis)
            first_0 = first_bbox_mode[0]
            last_1 = last_bbox_mode[1]
            
            return first_0, last_1
            
        top, bottom = locate_axis_boundary(table_df, 0, words)    
        left, right = locate_axis_boundary(table_df, 1, words)
        
        if any(val is None for val in [top, bottom, left, right]):
            return None
        
        refined_bbox = (left, top, right, bottom)

        return refined_bbox


def display_table_box(pdf_file, table_bbox, page_num=0, plumber_ordinate=1):
    """
    Display the part of the PDF corresponding to the table's bounding box.

    Args:
    - pdf_file (str): Path to the PDF file.
    - table_bbox (tuple): Bounding box of the table in the form of (x0, y0, x1, y1).
    - page_num (int): The page number where the table is located (default is 0).
    """
    # no need to convert, as the table_bbox is the pdfplumber version of adjusted bbox
    table_bbox_list = list(table_bbox)
    with pdfplumber.open(pdf_file) as pdf:
        page = pdf.pages[page_num]
        page_height = page.height  
        if plumber_ordinate!=1:
            table_bbox_list[1] = page_height - table_bbox[3]
            table_bbox_list[3] = page_height - table_bbox[1]
        cropped_page = page.within_bbox(table_bbox_list)
        im = cropped_page.to_image()
        plt.imshow(im.original)
        plt.axis('off')  
        plt.show()
        
def save_tables_to_xlsx(df_dict_list, output_path, start_idx):
    def sanitize_sheet_name(name, page_num):
    # Replace invalid characters with underscores and truncate to 31 characters
        name = re.sub(r'[\\/:*?"<>|]', '_', name)  
        limit = 31 - len(str(page_num)) - 1
        name = name[:limit] + '_' + str(page_num)
        return name
    df_dict_list = [df_dict for df_dict in df_dict_list if not df_dict['df'].empty]
    print(f'\nFinal num of df: {len(df_dict_list)}')
    # Use ExcelWriter to write multiple sheets
    with pd.ExcelWriter(output_path) as writer:
        for idx, df_dict in enumerate(df_dict_list):
            # Get the title or use "Sheet" + number if no title is provided
            page_num = df_dict['page']
            sheet_idx = start_idx + idx
            sheet_name = df_dict['title'] if df_dict['title'] else f'Sheet{sheet_idx + 1}'
            sheet_name = sanitize_sheet_name(sheet_name, page_num)
            df_dict['df'].to_excel(writer, sheet_name=sheet_name, index=False)

    print(f"Data successfully saved to {output_path}")
    

# return the order of column names if there is two-line column names after obtaining the cleaned table
def adjust_column_name_order(plumber_bbox, cleaned_table_df, pdf_path, page_num=0):
    if not plumber_bbox:
        return pd.DataFrame()
    cols = cleaned_table_df.columns.tolist()
    if not any(["_" in str(col) for col in cols]):
        return cleaned_table_df
    # has two-line column names
    print('Adjust column name order..')
    adjusted_table_df = cleaned_table_df.copy()
    # use unique words only for each col (compared with other column names)
    col_word_sets = {col: col.replace('_', " ").split() for col in cols}  
    unique_col_words = {}
    for col, word_set in col_word_sets.items():
        # Get words that are in this column but not in any other column
        other_column_words = set(word for c in cols for word in c.replace('_', " ").split() if c!=col)
        unique_words = [word for word in word_set if word not in other_column_words]
        if unique_words:
            unique_col_words[col] = unique_words
        else:
            unique_col_words[col] = word_set
    # collect the position of all col names and order 
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_num]  
        x0_table, y0_table, x1_table, y1_table = plumber_bbox
        # in case x0 is not accurate that lose column name information
        x0_table += -3
        # need clean the page.extract_words ['text'] -- different handling method for upper number
        covered_words = [
            word for word in page.extract_words()
            if x0_table <= word['x0'] <= x1_table 
            and y0_table-10 <= word['doctop'] <= y1_table 
            and abs(word['top'] - y0_table) <= 20  
        ]
        def remove_single_number(input_string):
            input_string = str(input_string)
            result = re.sub(r'(?<=\w)\d(?=\w)', '', input_string)  # Remove single number between letters
            return result
        covered_words = list(map(lambda word: {**word, 'text': remove_single_number(word['text'])}, covered_words))
        words = [
            word for word in covered_words
            if any(word['text'] in word_set for word_set in unique_col_words.values())  
        ]
        # sorted by x0
        words_sorted = sorted(words, key=lambda w: (w['x0'], w['doctop']))
        reordered_columns = []
        used_cols = set() # avoid repeatedness
        used_x0 = set()
        for word in words_sorted:
            best_match = None
            best_match_count = 0
            for col, word_set in unique_col_words.items():
                if col not in used_cols and word['x0'] not in used_x0:
                    match_count = sum(1 for col_word in word_set if col_word in word['text'])
                    if match_count > best_match_count:
                        best_match = col
                        best_match_count = match_count
            if best_match:
                reordered_columns.append(best_match)
                used_cols.add(best_match)
                used_x0.add(word['x0'])
        # Ensure all original columns are included (in case some were missed in extraction)
        missing_cols = [col for col in cols if col not in used_cols]
        reordered_columns.extend(missing_cols)
        adjusted_table_df.columns = reordered_columns
        print(f'adjusted column names: \n{reordered_columns}')
        
    return adjusted_table_df


if __name__ == "__main__":
    # test 
    pdf_path = "./data/SR/Totalenergies_2024.pdf"
    # pdf_path = "./data/table/Totalenergies_2024 (dragged) 2.pdf"
    df_dict_list = detect_extract_tables(pdf_path, save_tables=True)
    
