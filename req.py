import pip

def upgrade_packages(requirements_file):
    #upgrade all the packages in the given requirements file skipping the ones that have dependency conflicts

  installed_packages = set()
  conflicting_packages = set()

  with open(requirements_file, "r") as f:
    for package in f:
      try:
        pip.main(["install", "--upgrade", package])
        installed_packages.add(package)
      except pip.exceptions.ResolutionImpossible as e:
        conflicting_packages.add(package)

  for package in conflicting_packages:
    # Retry installing the conflicting package.
    try:
      pip.main(["install", "--upgrade", package])
      installed_packages.add(package)
    except pip.exceptions.ResolutionImpossible:
      # The package could not be installed because of a dependency conflict.
      # Skip the package and continue.
      pass

if __name__ == "__main__":
  requirements_file = "requirements.txt"
  upgrade_packages(requirements_file)
