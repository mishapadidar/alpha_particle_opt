def writePVD(outfileName,fileNames):
    outFile = open(outfileName, 'w')
    import xml.dom.minidom

    pvd = xml.dom.minidom.Document()
    pvd_root = pvd.createElementNS("VTK", "VTKFile")
    pvd_root.setAttribute("type", "Collection")
    pvd_root.setAttribute("version", "0.1")
    pvd_root.setAttribute("byte_order", "LittleEndian")
    pvd.appendChild(pvd_root)

    collection = pvd.createElementNS("VTK", "Collection")
    pvd_root.appendChild(collection)

    for i in range(len(fileNames)):
        dataSet = pvd.createElementNS("VTK", "DataSet")
        dataSet.setAttribute("timestep", str(i))
        dataSet.setAttribute("group", "")
        dataSet.setAttribute("part", "0")
        dataSet.setAttribute("file", str(fileNames[i]))
        collection.appendChild(dataSet)

    outFile = open(outfileName, 'w')
    pvd.writexml(outFile, newl='\n')
    outFile.close()

if __name__=="__main__":
  import glob
  import sys
  dirname = sys.argv[1]
  filenames = glob.glob(dirname + "/*.vts")
  writePVD(dirname + "/u_spatial_marginal.pvd",filenames)
  #filenames = glob.glob(dirname + "plot_data/*.vts")
  #writePVD("./u_spatial_marginal.pvd",filenames)
