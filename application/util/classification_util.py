def write_classification_to_file(classification_path, file_name, classification):
    f = open(classification_path + '/' + file_name + '.txt', "w")
    f.write(classification)
    f.close()
