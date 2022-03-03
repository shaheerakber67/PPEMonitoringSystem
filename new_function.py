import os
import cv2
import random
import numpy as np
import shutil
import pandas as pd
import calendar
from datetime import datetime
from fpdf import FPDF
import glob
from os import listdir
from absl import app, flags, logging

# function for creating report
data = [['ID','Class Name'],
        [0,'with_mask'],
        [1,'without_mask'],
        [2,'with_gloves'],
        [3,'without_gloves'],
        [4,'with_labcoat'],
        [5,'without_labcoat'],
        ]
path = "H:/NEW_FYP_project/Vision Based PPE Monitoring System (Final Year Project)/violations/" # get the path of images
class PDF(FPDF):

    def __init__(self):
        super().__init__()
        self.WIDTH = 210
        self.HEIGHT = 297
        
        
    def header(self):
        # header image
        self.image('H:/NEW_FYP_project/Vision Based PPE Monitoring System (Final Year Project)/assets/header.png', 0, 0, 210,45)
        # Line break
        self.ln(50)

        
    def footer(self):
        # Position at 1.5 cm from bottom
        self.set_y(-9)
        # Arial italic 8
        self.set_font('Arial', 'B', 8)
        self.image('H:/NEW_FYP_project/Vision Based PPE Monitoring System (Final Year Project)/assets/footer.png', 0, 255, 210,45)
        # Page number
        self.cell(0, 10, 'Page ' + str(self.page_no()) + '/{nb}', 0, 0, 'C')

    def page_body(self, images, data):
 
         # Since we do not need to draw lines anymore, there is no need to separate
         # headers from data matrix.
 
        # Effective page width, or just epw
        epw = pdf.w - 2* pdf.l_margin
         
        # Set column width to 1/4 of effective page width to distribute content 
        # evenly across table and page
        col_width = epw/2
         
        # Since we do not need to draw lines anymore, there is no need to separate
         # headers from data matrix.

        # Document title centered, 'B'old, 14 pt
        self.set_font('Times','B',14.0)
        self.cell(epw, 0.0, 'PPE Violations Report!')
        self.set_font('Times','',12.0) 
        self.ln(10)
        self.cell(epw, 0.0, 'Following are the use cases of PPEs which are used in the monitoring system and the violations reported:' )
        self.ln(10)
        self.image('H:/NEW_FYP_project/Vision Based PPE Monitoring System (Final Year Project)/assets/ppe.jpg', 60, 145, 100,100)
                # Text height is the same as current font size
        th = pdf.font_size
         
                # Here we add more padding by passing 2*th as height
        for row in data:
            for datum in row:
                # Enter data in colums
                pdf.cell(col_width, 2*th, str(datum), align='C', border=1 )
         
            self.ln(2*th)
        self.ln(10)


        imagelist = listdir(path) # get list of all images
        print(imagelist)

        x,y,w,h = 40, 60,120,120

        for image in imagelist:
            self.add_page()
            self.image(path+image,x,y,w,h)
            self.ln(130)
            self.cell(65)
            self.cell(50, 10, '' +str(image), 1, 1, 'C')
                
    def print_page(images,data):
        # Generates the report
        pdf.add_page()
        pdf.page_body(images,data)
        pdf.alias_nb_pages()
        pdf.output('ViolationReport.pdf','F')
     
        removing_files = glob.glob('H:/NEW_FYP_project/Vision Based PPE Monitoring System (Final Year Project)/violations/*.png')
        for i in removing_files:
            os.remove(i)

pdf = PDF()

# Remember to always put one of these at least once.
pdf.set_font('Times','',10.0) 



if __name__ == "__main__":
    
    pdf.print_page(path,data)   
    