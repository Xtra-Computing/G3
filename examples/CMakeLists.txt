# Gunrock's Testing CMake File

# begin /* Add premitives' subdirectories */

#add_subdirectory(graphsum)
#add_subdirectory(gtc)
#add_subdirectory(sparseMatMul)
#add_subdirectory(ReLU)
#add_subdirectory(MatMul)
#add_subdirectory(CrossEntropyLoss)
add_subdirectory(gcn)
#add_subdirectory(sgc)

# end /* Add premitives' subdirectories */

# note: Some premitives are not added as test because they don't have
#	cpu reference code.

if(GUNROCK_MGPU_TESTS)
  if(DEFINED DEVICES)
    set(DEVICES "--device=${DEVICES}")
  else(DEFINED DEVICES)
    message(WARNING "GUNROCK_MGPU_TESTS was set ON, but devices were not"
		    " specified, using `-DDEVICES=0,0` instead (default).")
    set(DEVICES "--device=0,0")
  endif(DEFINED DEVICES)
endif(GUNROCK_MGPU_TESTS)
# end /* Enable Testing for `ctest` */
