import webbrowser, os
import json
import ntpath
import boto3
import codecs
import io
from io import BytesIO
import sys
from pprint import pprint


def get_rows_columns_map(table_result, blocks_map):
    rows = {}
    for relationship in table_result['Relationships']:
        if relationship['Type'] == 'CHILD':
            for child_id in relationship['Ids']:
                cell = blocks_map[child_id]
                if cell['BlockType'] == 'CELL':
                    row_index = cell['RowIndex']
                    col_index = cell['ColumnIndex']
                    if row_index not in rows:
                        # create new row
                        rows[row_index] = {}
                        
                    # get the text value
                    rows[row_index][col_index] = get_text(cell, blocks_map)
    return rows


def get_text(result, blocks_map):
    text = ''
    if 'Relationships' in result:
        for relationship in result['Relationships']:
            if relationship['Type'] == 'CHILD':
                for child_id in relationship['Ids']:
                    word = blocks_map[child_id]
                    if word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    if word['BlockType'] == 'SELECTION_ELEMENT':
                        if word['SelectionStatus'] =='SELECTED':
                            text +=  'X '    
    return text


def get_table_csv_results(file_name):
    with open(file_name, 'rb') as file:
        img_test = file.read()
        bytes_test = bytearray(img_test)
        print('Image loaded', file_name)
    # process using image bytes
    # get the results
    client = boto3.client('textract')
    response = client.analyze_document(Document={'Bytes': bytes_test}, FeatureTypes=['TABLES'])
    # Get the text blocks
    blocks=response['Blocks']
    # pprint(blocks)
    blocks_map = {}
    table_blocks = []
    for block in blocks:
        blocks_map[block['Id']] = block
        if block['BlockType'] == "TABLE":
            table_blocks.append(block)
    if len(table_blocks) <= 0:
        return "<b> NO Table FOUND </b>"
    csv = ''
    for index, table in enumerate(table_blocks):
        csv += generate_table_csv(table, blocks_map, index +1)
        csv += '\n\n'

    return csv

def generate_table_csv(table_result, blocks_map, table_index):
    rows = get_rows_columns_map(table_result, blocks_map)
    table_id = 'Table_' + str(table_index)
    # get cells.
    # csv = 'Table: {0}\n\n'.format(table_id)
    csv = ''
    for row_index, cols in rows.items():
        
        for col_index, text in cols.items():
            csv += '{}'.format(text) + "\t"
        csv += '\n'    
    csv += '\n\n\n'
    return csv

def get_csv_from_image(file_path):
    file_name = ntpath.basename(file_path)
    table_csv = get_table_csv_results(file_path)
    output_file = os.path.join('C:/Users/zeto/Desktop/MasterThesis/project/amazon_csvs',file_name.replace('png','txt'))
    # replace content
    with codecs.open(output_file, 'w', encoding='utf8')as fout:
        fout.write(table_csv)
    # show the results
    return output_file


