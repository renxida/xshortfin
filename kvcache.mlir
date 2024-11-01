module @module {
  func.func @kvcache_ops(%arg0: !torch.tensor<[?,2662400],f16>, %arg1: !torch.vtensor<[16,32,100],f16>, %arg2: !torch.vtensor<[],si64>, %arg3: !torch.vtensor<[],si64>) -> !torch.vtensor<[2,16,32,100],f16> attributes {torch.assume_strict_symbolic_shapes} {
    %0 = torch.copy.to_vtensor %arg0 : !torch.vtensor<[?,2662400],f16>
    %int0 = torch.constant.int 0
    %1 = torch.aten.size.int %0, %int0 : !torch.vtensor<[?,2662400],f16>, !torch.int -> !torch.int
    %int26 = torch.constant.int 26
    %int2 = torch.constant.int 2
    %int16 = torch.constant.int 16
    %int32 = torch.constant.int 32
    %int100 = torch.constant.int 100
    %2 = torch.prim.ListConstruct %1, %int26, %int2, %int16, %int32, %int100 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %3 = torch.aten.view %0, %2 : !torch.vtensor<[?,2662400],f16>, !torch.list<int> -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int0_0 = torch.constant.int 0
    %4 = torch.aten.unsqueeze %arg2, %int0_0 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4 = torch.constant.int 4
    %5 = torch.prims.convert_element_type %4, %int4 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int0_1 = torch.constant.int 0
    %6 = torch.aten.unsqueeze %arg3, %int0_1 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_2 = torch.constant.int 4
    %7 = torch.prims.convert_element_type %6, %int4_2 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int2_3 = torch.constant.int 2
    %int0_4 = torch.constant.int 0
    %int9223372036854775807 = torch.constant.int 9223372036854775807
    %int1 = torch.constant.int 1
    %8 = torch.aten.slice.Tensor %3, %int2_3, %int0_4, %int9223372036854775807, %int1 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int3 = torch.constant.int 3
    %int0_5 = torch.constant.int 0
    %int9223372036854775807_6 = torch.constant.int 9223372036854775807
    %int1_7 = torch.constant.int 1
    %9 = torch.aten.slice.Tensor %8, %int3, %int0_5, %int9223372036854775807_6, %int1_7 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int4_8 = torch.constant.int 4
    %int0_9 = torch.constant.int 0
    %int9223372036854775807_10 = torch.constant.int 9223372036854775807
    %int1_11 = torch.constant.int 1
    %10 = torch.aten.slice.Tensor %9, %int4_8, %int0_9, %int9223372036854775807_10, %int1_11 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int5 = torch.constant.int 5
    %int0_12 = torch.constant.int 0
    %int9223372036854775807_13 = torch.constant.int 9223372036854775807
    %int1_14 = torch.constant.int 1
    %11 = torch.aten.slice.Tensor %10, %int5, %int0_12, %int9223372036854775807_13, %int1_14 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %12 = torch.prim.ListConstruct %5, %7 : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>
    %false = torch.constant.bool false
    %13 = torch.aten.index_put %11, %12, %arg1, %false : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<optional<vtensor>>, !torch.vtensor<[16,32,100],f16>, !torch.bool -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int26_15 = torch.constant.int 26
    %int2_16 = torch.constant.int 2
    %int16_17 = torch.constant.int 16
    %int32_18 = torch.constant.int 32
    %int100_19 = torch.constant.int 100
    %14 = torch.prim.ListConstruct %1, %int26_15, %int2_16, %int16_17, %int32_18, %int100_19 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %15 = torch.aten.view %0, %14 : !torch.vtensor<[?,2662400],f16>, !torch.list<int> -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int2_20 = torch.constant.int 2
    %int0_21 = torch.constant.int 0
    %int9223372036854775807_22 = torch.constant.int 9223372036854775807
    %int1_23 = torch.constant.int 1
    %16 = torch.aten.slice.Tensor %15, %int2_20, %int0_21, %int9223372036854775807_22, %int1_23 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int3_24 = torch.constant.int 3
    %int0_25 = torch.constant.int 0
    %int9223372036854775807_26 = torch.constant.int 9223372036854775807
    %int1_27 = torch.constant.int 1
    %17 = torch.aten.slice.Tensor %16, %int3_24, %int0_25, %int9223372036854775807_26, %int1_27 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int4_28 = torch.constant.int 4
    %int0_29 = torch.constant.int 0
    %int9223372036854775807_30 = torch.constant.int 9223372036854775807
    %int1_31 = torch.constant.int 1
    %18 = torch.aten.slice.Tensor %17, %int4_28, %int0_29, %int9223372036854775807_30, %int1_31 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int5_32 = torch.constant.int 5
    %int0_33 = torch.constant.int 0
    %int9223372036854775807_34 = torch.constant.int 9223372036854775807
    %int1_35 = torch.constant.int 1
    %19 = torch.aten.slice_scatter %18, %13, %int5_32, %int0_33, %int9223372036854775807_34, %int1_35 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int4_36 = torch.constant.int 4
    %int0_37 = torch.constant.int 0
    %int9223372036854775807_38 = torch.constant.int 9223372036854775807
    %int1_39 = torch.constant.int 1
    %20 = torch.aten.slice_scatter %17, %19, %int4_36, %int0_37, %int9223372036854775807_38, %int1_39 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int3_40 = torch.constant.int 3
    %int0_41 = torch.constant.int 0
    %int9223372036854775807_42 = torch.constant.int 9223372036854775807
    %int1_43 = torch.constant.int 1
    %21 = torch.aten.slice_scatter %16, %20, %int3_40, %int0_41, %int9223372036854775807_42, %int1_43 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int2_44 = torch.constant.int 2
    %int0_45 = torch.constant.int 0
    %int9223372036854775807_46 = torch.constant.int 9223372036854775807
    %int1_47 = torch.constant.int 1
    %22 = torch.aten.slice_scatter %15, %21, %int2_44, %int0_45, %int9223372036854775807_46, %int1_47 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int2662400 = torch.constant.int 2662400
    %23 = torch.prim.ListConstruct %1, %int2662400 : (!torch.int, !torch.int) -> !torch.list<int>
    %24 = torch.aten.view %22, %23 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<int> -> !torch.vtensor<[?,2662400],f16>
    torch.overwrite.tensor.contents %24 overwrites %arg0 : !torch.vtensor<[?,2662400],f16>, !torch.tensor<[?,2662400],f16>
    %int0_48 = torch.constant.int 0
    %25 = torch.aten.unsqueeze %arg2, %int0_48 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_49 = torch.constant.int 4
    %26 = torch.prims.convert_element_type %25, %int4_49 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int0_50 = torch.constant.int 0
    %27 = torch.aten.unsqueeze %arg3, %int0_50 : !torch.vtensor<[],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int4_51 = torch.constant.int 4
    %28 = torch.prims.convert_element_type %27, %int4_51 : !torch.vtensor<[1],si64>, !torch.int -> !torch.vtensor<[1],si64>
    %int26_52 = torch.constant.int 26
    %int2_53 = torch.constant.int 2
    %int16_54 = torch.constant.int 16
    %int32_55 = torch.constant.int 32
    %int100_56 = torch.constant.int 100
    %29 = torch.prim.ListConstruct %1, %int26_52, %int2_53, %int16_54, %int32_55, %int100_56 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %30 = torch.aten.view %24, %29 : !torch.vtensor<[?,2662400],f16>, !torch.list<int> -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int-1 = torch.constant.int -1
    %31 = torch.prim.ListConstruct %1, %int-1 : (!torch.int, !torch.int) -> !torch.list<int>
    %32 = torch.aten.view %30, %31 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<int> -> !torch.vtensor<[?,2662400],f16>
    %int26_57 = torch.constant.int 26
    %int2_58 = torch.constant.int 2
    %int16_59 = torch.constant.int 16
    %int32_60 = torch.constant.int 32
    %int100_61 = torch.constant.int 100
    %33 = torch.prim.ListConstruct %1, %int26_57, %int2_58, %int16_59, %int32_60, %int100_61 : (!torch.int, !torch.int, !torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
    %34 = torch.aten.view %32, %33 : !torch.vtensor<[?,2662400],f16>, !torch.list<int> -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int2_62 = torch.constant.int 2
    %int0_63 = torch.constant.int 0
    %int9223372036854775807_64 = torch.constant.int 9223372036854775807
    %int1_65 = torch.constant.int 1
    %35 = torch.aten.slice.Tensor %34, %int2_62, %:qint0_63, %int9223372036854775807_64, %int1_65 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int3_66 = torch.constant.int 3
    %int0_67 = torch.constant.int 0
    %int9223372036854775807_68 = torch.constant.int 9223372036854775807
    %int1_69 = torch.constant.int 1
    %36 = torch.aten.slice.Tensor %35, %int3_66, %int0_67, %int9223372036854775807_68, %int1_69 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int4_70 = torch.constant.int 4
    %int0_71 = torch.constant.int 0
    %int9223372036854775807_72 = torch.constant.int 9223372036854775807
    %int1_73 = torch.constant.int 1
    %37 = torch.aten.slice.Tensor %36, %int4_70, %int0_71, %int9223372036854775807_72, %int1_73 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %int5_74 = torch.constant.int 5
    %int0_75 = torch.constant.int 0
    %int9223372036854775807_76 = torch.constant.int 9223372036854775807
    %int1_77 = torch.constant.int 1
    %38 = torch.aten.slice.Tensor %37, %int5_74, %int0_75, %int9223372036854775807_76, %int1_77 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.int, !torch.int, !torch.int, !torch.int -> !torch.vtensor<[?,26,2,16,32,100],f16>
    %39 = torch.prim.ListConstruct %26, %28 : (!torch.vtensor<[1],si64>, !torch.vtensor<[1],si64>) -> !torch.list<optional<vtensor>>
    %40 = torch.aten.index.Tensor %38, %39 : !torch.vtensor<[?,26,2,16,32,100],f16>, !torch.list<optional<vtensor>> -> !torch.vtensor<[1,2,16,32,100],f16>
    %int0_78 = torch.constant.int 0
    %41 = torch.aten.squeeze.dim %40, %int0_78 : !torch.vtensor<[1,2,16,32,100],f16>, !torch.int -> !torch.vtensor<[2,16,32,100],f16>
    return %41 : !torch.vtensor<[2,16,32,100],f16>
  }
}
