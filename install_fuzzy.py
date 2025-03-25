import os
import subprocess
import sys

def install_fuzzy():
    print("Installing build dependencies for fuzzy...")
    
    # Install build essentials
    subprocess.check_call([sys.executable, "-m", "pip", "install", "setuptools", "wheel", "cython"])
    
    # Try direct install first
    try:
        print("Attempting to install fuzzy directly...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzy"])
        print("Fuzzy installed successfully!")
        return True
    except Exception as e:
        print(f"Direct installation failed: {e}")
    
    # Try installing from source
    try:
        print("Attempting to install fuzzy from source...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "git+https://github.com/yougov/fuzzy.git"])
        print("Fuzzy installed successfully from source!")
        return True
    except Exception as e:
        print(f"Source installation failed: {e}")
    
    # Try alternative package
    print("Attempting to install alternative package...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "double-metaphone"])
    print("Installed double-metaphone as an alternative to fuzzy")
    
    # Create a wrapper for compatibility
    print("Creating fuzzy compatibility wrapper...")
    wrapper_code = """
# Compatibility wrapper for fuzzy
class DMetaphone:
    def __call__(self, word):
        from double_metaphone import double_metaphone
        result = double_metaphone(word)
        # Convert to bytes to match fuzzy's output format
        return [result.encode('utf-8')] if result else [None]

class FuzzyModule:
    def __init__(self):
        self.DMetaphone = DMetaphone

# Create global instance
fuzzy = FuzzyModule()
"""
    
    # Write wrapper to a file
    with open("fuzzy_wrapper.py", "w") as f:
        f.write(wrapper_code.strip())
    
    print("Created fuzzy_wrapper.py - import this instead of fuzzy")
    return False

if __name__ == "__main__":
    success = install_fuzzy()
    if success:
        print("You can now import fuzzy normally")
    else:
        print("Use 'from fuzzy_wrapper import fuzzy' instead of 'import fuzzy'")
