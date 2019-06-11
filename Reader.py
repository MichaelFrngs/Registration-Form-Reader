import os
import pandas as pd
import glob
import PyPDF2

pdfFiles = []
def scanFolder4Pdfs():
    #enter folder to scan
    os.chdir(glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/')[0])
    #scan for pdfs
    for filename in os.listdir('.'):
        if filename.endswith('.pdf'):
            pdfFiles.append(filename)
    #Reset directory
    os.chdir("//busdata/Execprog/Executive Education Credit Programs/Student Records") 
    

#set directory to student records
os.chdir("C:/Users/mfrangos2016/Desktop/R/Leap Ahead Data Merger/RegFormChecker")

#List of names of the students: last, first
NamesData = pd.read_csv("names2.csv",header=None,index_col=None )

os.chdir("//busdata/Execprog/Executive Education Credit Programs/Student Records")


#Did we find the folder? Compare the results here with the regform section to see which ones are truly missing
for studentName in NamesData[0]:
    filename = studentName
    #* is wildcard. Verifies if the file exists. Returns true or false. Figure out a way to catch all registration forms.     
    FolderExists = glob.glob(f"//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{studentName}*")
    if FolderExists:
        pass
        print("Yes.", filename, "'s folder exists")
    else:
        print("ERROR",filename,"'s folder does not exist") 


SpringRegForm = ["*19s*reg*","*reg*19s*","*reg*1901*","*1901*reg*","*spring*19*reg*", "*19*spring*reg", "*reg*spring*19*","*reg*19*spring*","*19*reg*spring*","*spring*reg*19*"]
SummerRegForm = ["*19R*reg*","*reg*19R*","*reg*1905*","*1905*reg*","*summer*19*reg*", "*19*summer*reg", "*reg*summer*19*","*reg*19*summer*","*19*reg*summer*","*summer*reg*19*"]
############SELECT THE SEMESTER FROM LINE 41###############################
semesterRegForm = SummerRegForm

#leave alone. This portion sets names.
if semesterRegForm == SummerRegForm:
    semesterName = "Summer"
elif semesterRegForm == SpringRegForm:
    semesterName = "Spring"
elif semesterRegForm == FallRegForm:
    semesterName = "Fall"
else:
    print('semester selection ERROR, see line 44')


results = []
#Did we find the registration form?
for index, row in NamesData.iterrows():
    student_lastName = row.values[0]
    student_FirstName= row.values[1]
    filename = f"{student_lastName}, {student_FirstName}"
    
    #* is wildcard. Verifies if the file exists. Returns true or false. Figure out a way to catch all registration forms.    
    exists = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[0]}"}')
    exists2 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[1]}"}')
    exists3 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[2]}"}')
    exists4 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[3]}"}')
    exists5 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[4]}"}')
    exists6 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[5]}"}')
    exists7 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[6]}"}')
    exists8 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[7]}"}')
    exists9 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[8]}"}')
    exists10 = glob.glob(f'//busdata/Execprog/Executive Education Credit Programs/Student Records/*/*{student_lastName}*{student_FirstName}*/{f"{semesterRegForm[9]}"}')
    try:
        pdfFiles.append(scanFolder4Pdfs())
    except: 
        print(f"Program could not scan folder for {filename}'s PDF")
    try:
        if exists:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #1")
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists2:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #2")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists3:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #3")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists4:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists5:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")            
        elif exists6:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")      
        elif exists7:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists8:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists9:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        elif exists10:
            #print("Yes.", filename, f"'s {semesterName} Reg. form exists | Name Variant #4")
            #scanFolder4Pdfs()
            print("Yes.", filename, f"'s {semesterName} Reg. form exists")
            results.append(f"Yes. {filename}'s Reg. Form exists")
        else:
            print("ERROR",filename,f"'s {semesterName} Reg. form does not exist or is misnamed")
            results.append(f"{filename}'s {semesterName} Reg. form does not exist or is misnamed")
    except:
        #print(f"Program could not find folder for {studentName}")
        #results.append(f"Program could not find folder for {studentName}")
        print(f"Can't Find folder for {studentName}")

#Let's print the list
os.chdir("C:/Users/mfrangos2016/Desktop/R/Leap Ahead Data Merger/RegFormChecker")
df = pd.DataFrame(results, columns=["colummn"])
df.to_csv('Outputlist.csv', index=False)



        

##GET ALL REG FORMS
import tempfile
from pdf2image import convert_from_path
import os
Dirpath = "//REDACTED" 
os.chdir(Dirpath)

pdfFiles = []
for filename in os.listdir('.'):
        if filename.endswith('.pdf'):
            pdfFiles.append(filename)
for filename in pdfFiles:
    #do stuff
    #What is the name of the file to be converted?
    filename = f'{Dirpath}\{filename}'
    #filename = f'{Dirpath}\Baukol, Travis_Reg Form_17F.pdf'
    print(filename)
    
    with tempfile.TemporaryDirectory() as path:
         images_from_path = convert_from_path(filename, output_folder=rf"{Dirpath}/output", last_page=1, first_page =0)
     
    base_filename  =  os.path.splitext(os.path.basename(filename))[0] + '.jpeg'     
    #WHERE TO SAVE IMAGES
    save_dir = f'{Dirpath}/Saved'
    print(save_dir)
    for page in images_from_path:
        page.save(os.path.join(save_dir, base_filename), 'JPEG')


      
#CONVERT PDF FILES TO JPEG TO PREP FOR MACHINE LEARNING 
#Directory
import tempfile
from pdf2image import convert_from_path

import os
Dirpath = "C:/Users/mfrangos2016/Desktop/pdf-to-jpeg/pdf2jpg-master" 
os.chdir(Dirpath)

#import json
#with open('config.json') as config_file:
#    data = json.load(config_file)


pdfFiles = []
for filename in os.listdir('.'):
        if filename.endswith('.pdf'):
            pdfFiles.append(filename)
for filename in pdfFiles:
    #do stuff
    #What is the name of the file to be converted?
    filename = f'{Dirpath}\{filename}'
    #filename = f'{Dirpath}\Baukol, Travis_Reg Form_17F.pdf'
    print(filename)
    
    with tempfile.TemporaryDirectory() as path:
         images_from_path = convert_from_path(filename, output_folder=rf"{Dirpath}/output", last_page=1, first_page =0)
     
    base_filename  =  os.path.splitext(os.path.basename(filename))[0] + '.jpeg'     
    #WHERE TO SAVE IMAGES
    save_dir = f'{Dirpath}/Saved'
    print(save_dir)
    for page in images_from_path:
        page.save(os.path.join(save_dir, base_filename), 'JPEG')
    
 
