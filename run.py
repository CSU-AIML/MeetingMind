import os
import streamlit.web.bootstrap

if __name__ == "__main__":
    # Run the setup script first
    os.system("python setup.py")
    
    # Run the Streamlit app
    os.system("streamlit run app.py --server.port 8080 --server.address 0.0.0.0")
