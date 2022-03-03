import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

def main():

	fromaddr = "shaheerakber181@gmail.com"
	toaddr = "shaheerakbar67@gmail.com"

	msg = MIMEMultipart()

	msg['From'] = fromaddr
	msg['To'] = toaddr
	msg['Subject'] = "PPE Violations Report Attached"

	body = "Dear Compliance, \n\nI enclosed PPE Violations Report. \n\nRegards, \nIT Manager"
	msg.attach(MIMEText(body, 'plain'))

	filename = "ViolationReport.pdf"
	attachment = open("H:/NEW_FYP_project/Vision Based PPE Monitoring System (Final Year Project)/ViolationReport.pdf", "rb")

	part = MIMEBase('application', 'octet-stream')
	part.set_payload((attachment).read())
	encoders.encode_base64(part)
	part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

	msg.attach(part)

	server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
	server.login(fromaddr, "panther1999@-;")
	text = msg.as_string()
	server.sendmail(fromaddr, toaddr, text)
	server.quit()

if __name__ == '__main__':
    main()