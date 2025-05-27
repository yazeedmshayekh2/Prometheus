import pyodbc
from typing import Dict, Any
import pandas as pd
import urllib3
import requests

# Disable SSL warnings for PDF downloads
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

class DatabaseConnection:
    def __init__(self, connection_string: str = None):
        # Default connection string if none provided
        if not connection_string:
            self.connection_string = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                "SERVER=172.16.15.161;"
                "DATABASE=InsuranceOnlinePortal;"
                "UID=aiuser;"
                "PWD=AIP@ss0rdSQL;"
                "TrustServerCertificate=yes;"
                "Encrypt=yes;"
                "Timeout=30;"
            )
        else:
            self.connection_string = connection_string
            
    def get_connection_string(self):
        connection_string = (
                "DRIVER={ODBC Driver 18 for SQL Server};"
                "SERVER=172.16.15.161;"
                "DATABASE=InsuranceOnlinePortal;"
                "UID=aiuser;"
                "PWD=AIP@ss0rdSQL;"
                "TrustServerCertificate=yes;"
                "Encrypt=yes;"
                "Timeout=30;"
            )
        return connection_string
    
    def get_active_policies(self, national_id: str) -> pd.DataFrame:
        """Fetch active policies for a given national ID and their family members if any
        
        Returns DataFrame with columns:
        - CompanyName: Insurance company name
        - Relation: Relationship to primary policy holder
        - PolicyNo: Policy number
        - StartDate, EndDate: Contract validity period
        - AnnualLimit: Policy annual limit
        - AreaofCover: Geographical coverage area
        - EmergencyTreatment: Emergency treatment coverage
        - PDFLink: Link to policy document
        - ContractID: Contract identifier
        - ID, ParentID: Internal IDs for relationship mapping
        - IndividualID: Unique individual identifier
        - Name: Policy holder name
        - NationalID: National ID number
        - MobileNumber: Contact number
        - Email: Contact email
        - Benefits: Policy benefits
        - PolicyStartDate: Individual policy start date
        - StaffNo: Staff number if applicable
        """
        try:
            with pyodbc.connect(self.connection_string) as conn:
                query = """
                SELECT
                    c.CompanyName,
                    p.Relation,
                    c.PolicyNo,
                    c.StartDate,
                    c.EndDate,
                    c.AnnualLimit,
                    c.AreaofCover,
                    c.EmergencyTreatment,
                    c.PDFLink,
                    c.ContractID,
                    p.ID,
                    p.ParentID,
                    p.IndividualID,
                    p.Name,
                    p.NationalID,
                    p.MobileNumber,
                    p.Email,
                    p.Benefits,
                    p.StartDate AS PolicyStartDate,
                    p.StaffNo
                FROM 
                    tblHPolicies p
                INNER JOIN 
                    tblHContracts c 
                ON 
                    p.ContractID = c.ID
                WHERE 
                    c.isDeleted = 0 
                    AND c.EndDate > GETDATE()
                    AND (
                        p.NationalID = ?
                        OR ParentID = (SELECT ID FROM tblHPolicies WHERE NationalID = ?)
                    )
                """
                return pd.read_sql(query, conn, params=(national_id, national_id))
                
        except Exception as e:
            print(f"Database error: {str(e)}")
            print("Available ODBC drivers:", pyodbc.drivers())
            return pd.DataFrame()

    def download_policy_pdf(self, pdf_link: str) -> bytes:
        """Download policy PDF from the URL"""
        try:
            if not pdf_link:
                return None
                
            # Download with SSL verification disabled (only for internal URLs)
            response = requests.get(
                pdf_link, 
                verify=False, 
                timeout=30,
                headers={'User-Agent': 'Mozilla/5.0'}
            )
            response.raise_for_status()
            
            # Verify it's a PDF
            if 'application/pdf' not in response.headers.get('content-type', '').lower():
                print(f"Warning: Content may not be PDF: {pdf_link}")
                
            return response.content
            
        except Exception as e:
            print(f"Error downloading PDF from {pdf_link}: {str(e)}")
            return None
            
    def get_family_members(self, national_id: str) -> pd.DataFrame:
        """Get all family members associated with the policy holder"""
        try:
            print(f"Getting family members for National ID: {national_id}")
            with pyodbc.connect(self.connection_string) as conn:
                query = """
                DECLARE @PrincipalNationalID NVARCHAR(50) = ? -- Replace with the principal's national ID

                SELECT DISTINCT
                    p.NationalID,
                    p.Name,
                    p.ContractID,
                    p.CardURL,
                    p.DOB as DateOfBirth,
                    c.CompanyName,
                    c.PolicyNo,
                    c.StartDate as ContractStart,
                    c.EndDate as ContractEnd,
                    c.AnnualLimit,
                    c.AreaofCover,
                    c.EmergencyTreatment,
                    c.PDFLink,
                    CASE
                        WHEN UPPER(p.Relation) LIKE '%SPOUSE%' THEN 1
                        WHEN UPPER(p.Relation) LIKE '%CHILD%' THEN 2
                        ELSE 3
                    END as RelationOrder
                FROM dbo.tblHPolicies p
                JOIN dbo.tblHContracts c ON p.ContractID = c.ID
                WHERE p.ParentID = (
                    SELECT ID
                    FROM dbo.tblHPolicies
                    WHERE NationalID = @PrincipalNationalID
                    AND ID = ParentID
                )
                AND c.isDeleted = 0
                AND c.EndDate > GETDATE()  -- Only show active contracts
                ORDER BY
                    RelationOrder,
                    p.Name;
                """
                print("Executing family members query...")
                df = pd.read_sql(query, conn, params=[national_id])
                print(f"Query returned {len(df)} rows")
                if len(df) > 0:
                    print(f"Columns: {list(df.columns)}")
                    print("Sample data:")
                    print(df.head())
                return df
                
        except Exception as e:
            print(f"Error getting family members: {str(e)}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()

    def get_policy_details(self, national_id: str) -> Dict[str, Any]:
        """Get policy details for a national ID"""
        query = """
        SELECT DISTINCT
            c.CompanyName,
            p.Relation,
            c.PolicyNo,
            c.StartDate,
            c.EndDate,
            c.AnnualLimit,
            c.AreaofCover,
            c.EmergencyTreatment,
            c.PDFLink,
            p.Name,
            p.NationalID,
            p.MobileNumber,
            p.Email,
            p.Benefits,
            p.StartDate AS PolicyStartDate,
            p.StaffNo
        FROM 
            tblHPolicies p
        INNER JOIN 
            tblHContracts c 
        ON 
            p.ContractID = c.ID
        WHERE 
            c.isDeleted = 0 
            AND c.EndDate > GETDATE()
            AND p.NationalID = ?
        ORDER BY 
            c.StartDate DESC
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                cursor = conn.cursor()
                cursor.execute(query, (national_id,))
                columns = [column[0] for column in cursor.description]
                results = cursor.fetchall()
                
                if not results:
                    return {"error": "No active policies found"}
                
                # Convert results to DataFrame for easier processing
                df = pd.DataFrame.from_records(results, columns=columns)
                
                # Format policies
                formatted_policies = []
                for _, policy in df.iterrows():
                    formatted_policy = {
                        'policy_number': str(policy['PolicyNo']).strip(),
                        'company_name': str(policy['CompanyName']).strip(),
                        'start_date': policy['StartDate'].strftime('%Y-%m-%d') if pd.notnull(policy['StartDate']) else None,
                        'end_date': policy['EndDate'].strftime('%Y-%m-%d') if pd.notnull(policy['EndDate']) else None,
                        'annual_limit': str(policy['AnnualLimit']).strip() if pd.notnull(policy['AnnualLimit']) else None,
                        'area_of_cover': str(policy['AreaofCover']).strip() if pd.notnull(policy['AreaofCover']) else None,
                        'emergency_treatment': str(policy['EmergencyTreatment']).strip() if pd.notnull(policy['EmergencyTreatment']) else None,
                        'pdf_link': str(policy['PDFLink']).strip() if pd.notnull(policy['PDFLink']) else None,
                        'benefits': policy['Benefits'] if pd.notnull(policy['Benefits']) else None
                    }
                    formatted_policies.append(formatted_policy)
                
                # Format primary member details
                primary_member = {
                    'name': str(df.iloc[0]['Name']).strip(),
                    'national_id': str(df.iloc[0]['NationalID']).strip(),
                    'relation': str(df.iloc[0]['Relation']).strip(),
                    'mobile': str(df.iloc[0]['MobileNumber']).strip() if pd.notnull(df.iloc[0]['MobileNumber']) else None,
                    'email': str(df.iloc[0]['Email']).strip() if pd.notnull(df.iloc[0]['Email']) else None,
                    'staff_no': str(df.iloc[0]['StaffNo']).strip() if pd.notnull(df.iloc[0]['StaffNo']) else None,
                    'policies': formatted_policies
                }
                
                return {"primary_member": primary_member}
                
        except Exception as e:
            print(f"Error in get_policy_details: {str(e)}")
            return {"error": str(e)}

    def get_all_policies(self):
        """Get all active and recent policy documents from database"""
        query = """
        SELECT DISTINCT 
            c.PDFLink as pdf_link,
            c.CompanyName as company_name,
            c.StartDate as start_date,
            c.EndDate as end_date
        FROM tblHContracts c
        WHERE 
            c.PDFLink IS NOT NULL
            AND c.isDeleted = 0 
            AND c.EndDate > GETDATE()  -- Active policies only
            AND c.StartDate >= '2020-01-01'  -- Recent policies only
            AND c.PDFLink LIKE '%TOB%'  -- TOB documents only
        ORDER BY 
            c.StartDate DESC  -- Get newest first
        """
        
        try:
            with pyodbc.connect(self.connection_string) as conn:
                df = pd.read_sql(query, conn)
                policies = df.to_dict('records')
                
                # Convert datetime objects to strings
                for policy in policies:
                    if pd.notnull(policy['start_date']):
                        policy['start_date'] = policy['start_date'].strftime('%Y-%m-%d')
                    if pd.notnull(policy['end_date']):
                        policy['end_date'] = policy['end_date'].strftime('%Y-%m-%d')
                        
                print(f"Found {len(policies)} active and recent TOB documents")
                return policies
                
        except Exception as e:
            print(f"Error in get_all_policies: {str(e)}")
            return []