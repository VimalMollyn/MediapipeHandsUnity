// <auto-generated>
//     Generated by the protocol buffer compiler.  DO NOT EDIT!
//     source: mediapipe/calculators/util/detection_label_id_to_text_calculator.proto
// </auto-generated>
#pragma warning disable 1591, 0612, 3021
#region Designer generated code

using pb = global::Google.Protobuf;
using pbc = global::Google.Protobuf.Collections;
using pbr = global::Google.Protobuf.Reflection;
using scg = global::System.Collections.Generic;
namespace Mediapipe {

  /// <summary>Holder for reflection information generated from mediapipe/calculators/util/detection_label_id_to_text_calculator.proto</summary>
  public static partial class DetectionLabelIdToTextCalculatorReflection {

    #region Descriptor
    /// <summary>File descriptor for mediapipe/calculators/util/detection_label_id_to_text_calculator.proto</summary>
    public static pbr::FileDescriptor Descriptor {
      get { return descriptor; }
    }
    private static pbr::FileDescriptor descriptor;

    static DetectionLabelIdToTextCalculatorReflection() {
      byte[] descriptorData = global::System.Convert.FromBase64String(
          string.Concat(
            "CkZtZWRpYXBpcGUvY2FsY3VsYXRvcnMvdXRpbC9kZXRlY3Rpb25fbGFiZWxf",
            "aWRfdG9fdGV4dF9jYWxjdWxhdG9yLnByb3RvEgltZWRpYXBpcGUaJG1lZGlh",
            "cGlwZS9mcmFtZXdvcmsvY2FsY3VsYXRvci5wcm90bxoebWVkaWFwaXBlL3V0",
            "aWwvbGFiZWxfbWFwLnByb3RvIvEBCidEZXRlY3Rpb25MYWJlbElkVG9UZXh0",
            "Q2FsY3VsYXRvck9wdGlvbnMSFgoObGFiZWxfbWFwX3BhdGgYASABKAkSDQoF",
            "bGFiZWwYAiADKAkSFQoNa2VlcF9sYWJlbF9pZBgDIAEoCBImCglsYWJlbF9t",
            "YXAYBCABKAsyEy5tZWRpYXBpcGUuTGFiZWxNYXAyYAoDZXh0EhwubWVkaWFw",
            "aXBlLkNhbGN1bGF0b3JPcHRpb25zGLCLjnggASgLMjIubWVkaWFwaXBlLkRl",
            "dGVjdGlvbkxhYmVsSWRUb1RleHRDYWxjdWxhdG9yT3B0aW9ucw=="));
      descriptor = pbr::FileDescriptor.FromGeneratedCode(descriptorData,
          new pbr::FileDescriptor[] { global::Mediapipe.CalculatorReflection.Descriptor, global::Mediapipe.LabelMapReflection.Descriptor, },
          new pbr::GeneratedClrTypeInfo(null, null, new pbr::GeneratedClrTypeInfo[] {
            new pbr::GeneratedClrTypeInfo(typeof(global::Mediapipe.DetectionLabelIdToTextCalculatorOptions), global::Mediapipe.DetectionLabelIdToTextCalculatorOptions.Parser, new[]{ "LabelMapPath", "Label", "KeepLabelId", "LabelMap" }, null, null, new pb::Extension[] { global::Mediapipe.DetectionLabelIdToTextCalculatorOptions.Extensions.Ext }, null)
          }));
    }
    #endregion

  }
  #region Messages
  public sealed partial class DetectionLabelIdToTextCalculatorOptions : pb::IMessage<DetectionLabelIdToTextCalculatorOptions>
  #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
      , pb::IBufferMessage
  #endif
  {
    private static readonly pb::MessageParser<DetectionLabelIdToTextCalculatorOptions> _parser = new pb::MessageParser<DetectionLabelIdToTextCalculatorOptions>(() => new DetectionLabelIdToTextCalculatorOptions());
    private pb::UnknownFieldSet _unknownFields;
    private int _hasBits0;
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public static pb::MessageParser<DetectionLabelIdToTextCalculatorOptions> Parser { get { return _parser; } }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public static pbr::MessageDescriptor Descriptor {
      get { return global::Mediapipe.DetectionLabelIdToTextCalculatorReflection.Descriptor.MessageTypes[0]; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    pbr::MessageDescriptor pb::IMessage.Descriptor {
      get { return Descriptor; }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public DetectionLabelIdToTextCalculatorOptions() {
      OnConstruction();
    }

    partial void OnConstruction();

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public DetectionLabelIdToTextCalculatorOptions(DetectionLabelIdToTextCalculatorOptions other) : this() {
      _hasBits0 = other._hasBits0;
      labelMapPath_ = other.labelMapPath_;
      label_ = other.label_.Clone();
      keepLabelId_ = other.keepLabelId_;
      labelMap_ = other.labelMap_ != null ? other.labelMap_.Clone() : null;
      _unknownFields = pb::UnknownFieldSet.Clone(other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public DetectionLabelIdToTextCalculatorOptions Clone() {
      return new DetectionLabelIdToTextCalculatorOptions(this);
    }

    /// <summary>Field number for the "label_map_path" field.</summary>
    public const int LabelMapPathFieldNumber = 1;
    private readonly static string LabelMapPathDefaultValue = "";

    private string labelMapPath_;
    /// <summary>
    /// Path to a label map file for getting the actual name of detected classes.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public string LabelMapPath {
      get { return labelMapPath_ ?? LabelMapPathDefaultValue; }
      set {
        labelMapPath_ = pb::ProtoPreconditions.CheckNotNull(value, "value");
      }
    }
    /// <summary>Gets whether the "label_map_path" field is set</summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public bool HasLabelMapPath {
      get { return labelMapPath_ != null; }
    }
    /// <summary>Clears the value of the "label_map_path" field</summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public void ClearLabelMapPath() {
      labelMapPath_ = null;
    }

    /// <summary>Field number for the "label" field.</summary>
    public const int LabelFieldNumber = 2;
    private static readonly pb::FieldCodec<string> _repeated_label_codec
        = pb::FieldCodec.ForString(18);
    private readonly pbc::RepeatedField<string> label_ = new pbc::RepeatedField<string>();
    /// <summary>
    /// Alternative way to specify label map.
    /// label: "label for id 0"
    /// label: "label for id 1"
    /// ...
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public pbc::RepeatedField<string> Label {
      get { return label_; }
    }

    /// <summary>Field number for the "keep_label_id" field.</summary>
    public const int KeepLabelIdFieldNumber = 3;
    private readonly static bool KeepLabelIdDefaultValue = false;

    private bool keepLabelId_;
    /// <summary>
    /// By default, the `label_id` field from the input is stripped if a text label
    /// could be found. By setting this field to true, it is always copied to the
    /// output detections.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public bool KeepLabelId {
      get { if ((_hasBits0 & 1) != 0) { return keepLabelId_; } else { return KeepLabelIdDefaultValue; } }
      set {
        _hasBits0 |= 1;
        keepLabelId_ = value;
      }
    }
    /// <summary>Gets whether the "keep_label_id" field is set</summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public bool HasKeepLabelId {
      get { return (_hasBits0 & 1) != 0; }
    }
    /// <summary>Clears the value of the "keep_label_id" field</summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public void ClearKeepLabelId() {
      _hasBits0 &= ~1;
    }

    /// <summary>Field number for the "label_map" field.</summary>
    public const int LabelMapFieldNumber = 4;
    private global::Mediapipe.LabelMap labelMap_;
    /// <summary>
    /// Label map.
    /// </summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public global::Mediapipe.LabelMap LabelMap {
      get { return labelMap_; }
      set {
        labelMap_ = value;
      }
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public override bool Equals(object other) {
      return Equals(other as DetectionLabelIdToTextCalculatorOptions);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public bool Equals(DetectionLabelIdToTextCalculatorOptions other) {
      if (ReferenceEquals(other, null)) {
        return false;
      }
      if (ReferenceEquals(other, this)) {
        return true;
      }
      if (LabelMapPath != other.LabelMapPath) return false;
      if(!label_.Equals(other.label_)) return false;
      if (KeepLabelId != other.KeepLabelId) return false;
      if (!object.Equals(LabelMap, other.LabelMap)) return false;
      return Equals(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public override int GetHashCode() {
      int hash = 1;
      if (HasLabelMapPath) hash ^= LabelMapPath.GetHashCode();
      hash ^= label_.GetHashCode();
      if (HasKeepLabelId) hash ^= KeepLabelId.GetHashCode();
      if (labelMap_ != null) hash ^= LabelMap.GetHashCode();
      if (_unknownFields != null) {
        hash ^= _unknownFields.GetHashCode();
      }
      return hash;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public override string ToString() {
      return pb::JsonFormatter.ToDiagnosticString(this);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public void WriteTo(pb::CodedOutputStream output) {
    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
      output.WriteRawMessage(this);
    #else
      if (HasLabelMapPath) {
        output.WriteRawTag(10);
        output.WriteString(LabelMapPath);
      }
      label_.WriteTo(output, _repeated_label_codec);
      if (HasKeepLabelId) {
        output.WriteRawTag(24);
        output.WriteBool(KeepLabelId);
      }
      if (labelMap_ != null) {
        output.WriteRawTag(34);
        output.WriteMessage(LabelMap);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(output);
      }
    #endif
    }

    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    void pb::IBufferMessage.InternalWriteTo(ref pb::WriteContext output) {
      if (HasLabelMapPath) {
        output.WriteRawTag(10);
        output.WriteString(LabelMapPath);
      }
      label_.WriteTo(ref output, _repeated_label_codec);
      if (HasKeepLabelId) {
        output.WriteRawTag(24);
        output.WriteBool(KeepLabelId);
      }
      if (labelMap_ != null) {
        output.WriteRawTag(34);
        output.WriteMessage(LabelMap);
      }
      if (_unknownFields != null) {
        _unknownFields.WriteTo(ref output);
      }
    }
    #endif

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public int CalculateSize() {
      int size = 0;
      if (HasLabelMapPath) {
        size += 1 + pb::CodedOutputStream.ComputeStringSize(LabelMapPath);
      }
      size += label_.CalculateSize(_repeated_label_codec);
      if (HasKeepLabelId) {
        size += 1 + 1;
      }
      if (labelMap_ != null) {
        size += 1 + pb::CodedOutputStream.ComputeMessageSize(LabelMap);
      }
      if (_unknownFields != null) {
        size += _unknownFields.CalculateSize();
      }
      return size;
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public void MergeFrom(DetectionLabelIdToTextCalculatorOptions other) {
      if (other == null) {
        return;
      }
      if (other.HasLabelMapPath) {
        LabelMapPath = other.LabelMapPath;
      }
      label_.Add(other.label_);
      if (other.HasKeepLabelId) {
        KeepLabelId = other.KeepLabelId;
      }
      if (other.labelMap_ != null) {
        if (labelMap_ == null) {
          LabelMap = new global::Mediapipe.LabelMap();
        }
        LabelMap.MergeFrom(other.LabelMap);
      }
      _unknownFields = pb::UnknownFieldSet.MergeFrom(_unknownFields, other._unknownFields);
    }

    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public void MergeFrom(pb::CodedInputStream input) {
    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
      input.ReadRawMessage(this);
    #else
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, input);
            break;
          case 10: {
            LabelMapPath = input.ReadString();
            break;
          }
          case 18: {
            label_.AddEntriesFrom(input, _repeated_label_codec);
            break;
          }
          case 24: {
            KeepLabelId = input.ReadBool();
            break;
          }
          case 34: {
            if (labelMap_ == null) {
              LabelMap = new global::Mediapipe.LabelMap();
            }
            input.ReadMessage(LabelMap);
            break;
          }
        }
      }
    #endif
    }

    #if !GOOGLE_PROTOBUF_REFSTRUCT_COMPATIBILITY_MODE
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    void pb::IBufferMessage.InternalMergeFrom(ref pb::ParseContext input) {
      uint tag;
      while ((tag = input.ReadTag()) != 0) {
        switch(tag) {
          default:
            _unknownFields = pb::UnknownFieldSet.MergeFieldFrom(_unknownFields, ref input);
            break;
          case 10: {
            LabelMapPath = input.ReadString();
            break;
          }
          case 18: {
            label_.AddEntriesFrom(ref input, _repeated_label_codec);
            break;
          }
          case 24: {
            KeepLabelId = input.ReadBool();
            break;
          }
          case 34: {
            if (labelMap_ == null) {
              LabelMap = new global::Mediapipe.LabelMap();
            }
            input.ReadMessage(LabelMap);
            break;
          }
        }
      }
    }
    #endif

    #region Extensions
    /// <summary>Container for extensions for other messages declared in the DetectionLabelIdToTextCalculatorOptions message type.</summary>
    [global::System.Diagnostics.DebuggerNonUserCodeAttribute]
    [global::System.CodeDom.Compiler.GeneratedCode("protoc", null)]
    public static partial class Extensions {
      public static readonly pb::Extension<global::Mediapipe.CalculatorOptions, global::Mediapipe.DetectionLabelIdToTextCalculatorOptions> Ext =
        new pb::Extension<global::Mediapipe.CalculatorOptions, global::Mediapipe.DetectionLabelIdToTextCalculatorOptions>(251889072, pb::FieldCodec.ForMessage(2015112578, global::Mediapipe.DetectionLabelIdToTextCalculatorOptions.Parser));
    }
    #endregion

  }

  #endregion

}

#endregion Designer generated code
