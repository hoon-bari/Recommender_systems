# config.py
RESUME_ALL_FIELDS = ['resume_seq', 'graduate_date', 'hope_salary', 'last_salary', 
                     'career_month', 'univ_score', 'update_reg_diff',
                     'language', 'degree', 'career_job_code', 
                     'certificate_count','job_code_seq1', 'job_code_seq2', 
                     'job_code_seq3', 'univ_transfer', 'univ_location', 
                     'univ_major_type', 'hischool_type_seq', 'hischool_special_type', 
                     'hischool_nation', 'hischool_gender', 'hischool_location_seq',
                     'univ_type_seq1', 'univ_type_seq2', 'text_keyword']

RESUME_CONT_FIELDS = ['resume_seq', 'graduate_date', 'hope_salary', 'last_salary', 
                      'career_month', 'univ_score', 'update_reg_diff']

RESUME_CAT_FIELDS = list(set(RESUME_ALL_FIELDS).difference(RESUME_CONT_FIELDS))


ALL_FIELDS = ['label', 'resume_seq', 'recruitment_seq', 'graduate_date', 
              'hope_salary', 'last_salary', 'career_month', 
              'univ_score', 'update_reg_diff', 'language', 
              'degree', 'career_job_code', 'certificate_count',
              'job_code_seq1', 'job_code_seq2', 'job_code_seq3', 
              'univ_transfer', 'univ_location', 'univ_major_type', 
              'hischool_type_seq', 'hischool_special_type', 'hischool_nation', 
              'hischool_gender', 'hischool_location_seq', 'univ_type_seq1', 
              'univ_type_seq2', 'text_keyword', 'employee', 'address_seq1', 
              'check_box_keyword', 'education', 'major_task', 
              'qualifications', 'company_type_seq', 'supply_kind']

CONT_FIELDS = ['resume_seq', 'recruitment_seq', 'graduate_date', 'hope_salary', 'last_salary', 
               'career_month', 'univ_score', 'update_reg_diff', 'employee']

CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))


# Hyper-parameters for Experiment
NUM_BIN = 10
BATCH_SIZE = 64
EMBEDDING_SIZE = 10
SEED = 42
EPOCHS = 40