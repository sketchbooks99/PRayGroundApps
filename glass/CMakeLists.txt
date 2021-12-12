PRAYGROUND_add_executalbe(glass target_name
    main.cpp 
    app.cpp 
    app.h
    params.h
    box_medium.h 
    box_medium.cpp 
    kernels.cu
)

target_compile_definitions(
    ${target_name}
    PRIVATE
    APP_DIR="${CMAKE_CURRENT_SOURCE_DIR}"
)

target_link_libraries(${target_name} ${CUDA_LIBRARIES})