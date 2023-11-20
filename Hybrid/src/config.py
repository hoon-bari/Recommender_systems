ALL_FIELDS = ['graduate_date', 'hope_salary', 'last_salary', 
              'univ_score', 'reg_year', 'reg_month', 'reg_day',
              'updated_year', 'updated_month', 'updated_day',
              'update_reg_diff', 'language', 'career_month', 
              'degree', 'career_job_code', 'certificate_count',
              'job_code_seq1', 'job_code_seq2', 'job_code_seq3', 
              'univ_transfer', 'univ_location', 'univ_major_type', 
              'hischool_type_seq', 'hischool_special_type', 'hischool_nation', 
              'hischool_gender', 'hischool_location_seq', 'univ_type_seq1', 
              'univ_type_seq2', 'text_keyword', 'employee', 'address_seq1', 
              'check_box_keyword', 'education', 'major_task', 
              'qualifications', 'company_type_seq', 'supply_kind']

CONT_FIELDS = ['graduate_date', 'hope_salary', 'last_salary',
               'reg_year', 'reg_month', 'reg_day',
                'updated_year', 'updated_month', 'updated_day',
               'career_month', 'univ_score', 'update_reg_diff', 'employee']

CAT_FIELDS = list(set(ALL_FIELDS).difference(CONT_FIELDS))