import tableExtraction
import tableProcess
import os

def table_to_json_pipeline(pdf_path, output_dir, save_tables=True, to_excel=False, to_json=True):
    file_name_with_extension = os.path.basename(pdf_path)
    file_name = os.path.splitext(file_name_with_extension)[0]
    sheets_dict, df_dict_list = tableExtraction.detect_extract_tables(pdf_path, save_tables=save_tables)
    # print(df_dict_list)
    sheets_dict_1 = tableProcess.tabledict_to_json(sheets_dict, file_name, to_excel=to_excel, to_json=to_json, output_dir=output_dir)
    return sheets_dict_1

if __name__ == '__main__':
    pdf_path = './data/SR/2023_longi-sustainability-report.pdf'
    output_dir = './output/table'
    sheets_dict_1 = table_to_json_pipeline(pdf_path, output_dir=output_dir, to_excel=True)