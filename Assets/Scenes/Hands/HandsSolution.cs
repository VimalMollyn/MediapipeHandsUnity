using System.Collections;
using UnityEngine;
using UnityEngine.UI;
using System.Collections.Generic;
using System.Linq;

using Stopwatch = System.Diagnostics.Stopwatch;

namespace Mediapipe.Unity.Tutorial
{
  public class HandsSolution : MonoBehaviour
  {
    [SerializeField] private TextAsset _configAssetCPU, _configAssetGPU;
    [SerializeField] private RawImage _screen;
    [SerializeField] private int _width;
    [SerializeField] private int _height;
    [SerializeField] private int _fps;

    private CalculatorGraph _graph;
    private ResourceManager _resourceManager;

    private WebCamTexture _webCamTexture;
    private Texture2D _inputTexture;
    private Color32[] _inputPixelData;
    private Texture2D _outputTexture;
    private Color32[] _outputPixelData;
    private bool hasGPU;

    private IEnumerator Start()
    {
      if (WebCamTexture.devices.Length == 0)
      {
        throw new System.Exception("Web Camera devices are not found");
      }

      var webCamDevice = WebCamTexture.devices[0];
      _webCamTexture = new WebCamTexture(webCamDevice.name, _width, _height, _fps);
      _webCamTexture.Play();


      yield return new WaitUntil(() => _webCamTexture.width > 16);
      yield return GpuManager.Initialize();
      
      // we requested a height and width, but now we update those vars with the actual width and height
      _width = _webCamTexture.width;
      _height = _webCamTexture.height;

      if (!GpuManager.IsInitialized)
      {
        Debug.Log("Failed to initialize GPU resources, falling back to CPU");
        hasGPU = false;
      }
      else {
        hasGPU = true;
      }

      _screen.rectTransform.sizeDelta = new Vector2(_width, _height);

      _inputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
      _inputPixelData = new Color32[_width * _height];
      _outputTexture = new Texture2D(_width, _height, TextureFormat.RGBA32, false);
      _outputPixelData = new Color32[_width * _height];

      _screen.texture = _outputTexture;

      AssetLoader.Provide(new StreamingAssetsResourceManager());
      yield return AssetLoader.PrepareAssetAsync("hand_landmark_lite.bytes", "hand_landmark_lite.bytes", false); 
      yield return AssetLoader.PrepareAssetAsync("handedness.txt", "handedness.txt", false);
      yield return AssetLoader.PrepareAssetAsync("palm_detection_lite.bytes", "palm_detection_lite.bytes", false);

      var stopwatch = new Stopwatch();

      if(hasGPU) {
        _graph = new CalculatorGraph(_configAssetGPU.text);
        _graph.SetGpuResources(GpuManager.GpuResources).AssertOk();
      }
      else {
        _graph = new CalculatorGraph(_configAssetCPU.text);
      }

      var poseLandmarkStream = new OutputStream<NormalizedLandmarkListVectorPacket, List<NormalizedLandmarkList>>(_graph, "hand_landmarks");
      poseLandmarkStream.StartPolling().AssertOk();
      _graph.StartRun().AssertOk();
      stopwatch.Start();

      while (true)
      {
        _inputTexture.SetPixels32(_webCamTexture.GetPixels32(_inputPixelData));
        var imageFrame = new ImageFrame(ImageFormat.Types.Format.Srgba, _width, _height, _width * 4, _inputTexture.GetRawTextureData<byte>());
        var currentTimestamp = stopwatch.ElapsedTicks / (System.TimeSpan.TicksPerMillisecond / 1000);
        _graph.AddPacketToInputStream("input_video", new ImageFramePacket(imageFrame, new Timestamp(currentTimestamp))).AssertOk();

        yield return new WaitForEndOfFrame();

        try {
            if (poseLandmarkStream.TryGetNext(out var poseLandmarks))
            {
              if (poseLandmarks != null)
              {
                // top of the head
                var nose = poseLandmarks[0].Landmark[0];
                // var screenPoint = screenRect.GetPoint(nose);
                var scale = 0.1545784f; // TODO: Not sure how to get this number programmatically
                // noseSphere.transform.position = new Vector3(-screenPoint.x * scale, screenPoint.y * scale, 90f);
                Debug.Log(nose.ToString());
              }
            }
        }
        catch (System.Exception e)
        {
            Debug.Log($"Some exception, {e}");
        }

      }
    }

    private void OnDestroy()
    {
      if (_webCamTexture != null)
      {
        _webCamTexture.Stop();
      }

      if (_graph != null)
      {
        try
        {
          _graph.CloseInputStream("input_video").AssertOk();
          _graph.WaitUntilDone().AssertOk();
        }
        finally
        {

          _graph.Dispose();
        }
      }

      if(hasGPU) { 
        GpuManager.Shutdown();
      }
    }
  }
}
