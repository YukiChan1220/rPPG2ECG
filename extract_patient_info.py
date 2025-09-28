from data.patient_info import PatientInfo

data_dir = "./patient_data"
output_file = "overall_patient_info.csv"
patient_info = PatientInfo(data_dir, save_dir=output_file)
patient_info_list = patient_info.extract()
patient_info.save()
