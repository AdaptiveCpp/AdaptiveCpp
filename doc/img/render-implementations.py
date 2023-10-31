from graphviz import Graph
from graphviz import Digraph
import colorsys
import math

#layout = "top-bottom"
layout = "left-right"
#display_mode = "shared-nodes"
display_mode = "no-hardware-nodes"
#display_mode = "no-shared-nodes"
outputformat = "png"

default_tailport="s"
default_headport="n"
if layout == "left-right":
  default_tailport="e"
  default_headport="w"

default_node_width="2.5"

class HSVColor:
  def __init__(self, H, S, V):
    self.h = H
    self.s = S
    self.v = V

  def __str__(self):
    return "{:.3f} {:.3f} {:.3f}".format(self.h, self.s, self.v)

  def perceived_brightness(self):
    rgb = colorsys.hsv_to_rgb(self.h, self.s, self.v)
    r = rgb[0]
    g = rgb[1]
    b = rgb[2]
    return math.sqrt(r*r*.241+g*g*.691+b*b*.068)

  @property
  def V(self):
    return self.v

  @property
  def S(self):
    return self.s

  @property
  def H(self):
    return self.h

def determine_font_color(bg_color):
  font_color = HSVColor(0.0, 0.0, 0.0)
  if bg_color.perceived_brightness() < 0.6:
    font_color = HSVColor(0.0, 0.0, 1.0)
  return font_color

def make_hsv_color(h,s,v):
  return HSVColor(h,s,v)

def make_rgb_color(r,g,b):
  hsv = colorsys.rgb_to_hsv(r, g, b)
  return HSVColor(hsv[0], hsv[1], hsv[2])

def make_html_color(color_string):

  mystr = str(color_string)
  if len(mystr) == 7:
    mystr = mystr[1:]

  if len(mystr) != 6:
    raise RuntimeError("Invalid html color")

  r = int(mystr[0:2], 16) / 255.
  g = int(mystr[2:4], 16) / 255.
  b = int(mystr[4: ], 16) / 255.

  return make_rgb_color(r,g,b)


footnote_number = 0
footnote_text = ""

def add_footnote(text):
  global footnote_number
  global footnote_text


  footnote_text += "[{}] {}\l".format(footnote_number, text)
  footnote_number += 1
  return "[{}]".format(footnote_number-1)

def add_backend(graph, id_name, color, supported_hardware_colors, description=None, device_caveats={}):

  if description == None:
    description = id_name

  graph.node(id_name, description, shape='box', color=str(color), width=default_node_width,
              style='rounded,filled', fontcolor = str(determine_font_color(color)))

  if display_mode == "no-hardware-nodes":
    hw_description = ""
    for hw in supported_hardware_colors:

      if hw in device_caveats:
        hw_description += "{} ({})\n".format(hw,device_caveats[hw])
      else:
        hw_description += "{}\n".format(hw)

    hw_node = id_name+"_supported_hw"
    graph.node(hw_node, hw_description, shape="plain", style='filled', color="#ffffff00", labelloc="t")
    graph.edge(id_name, hw_node, style='invis')
  else:
    for hw in supported_hardware_colors:
      # construct unique id for hardware node if requested in display_mode.
      # This controls whether multiple implementations targeting the same hardware
      # are mapped to the same, or a duplicated hardware node.
      hw_id = hw+id_name
      if display_mode == "shared-nodes":
        hw_id = hw

      graph.node(hw_id, hw, shape='box', color=str(supported_hardware_colors[hw]),
                style='rounded,filled', width=default_node_width, fontcolor = str(determine_font_color(supported_hardware_colors[hw])))
      if hw in device_caveats:
        graph.edge(id_name, hw_id, penwidth='2',
                  color=str(color), style='dashed', labelfloat="true", labeldistance="2.0",
                  headlabel=add_footnote(device_caveats[hw]))
      else:
        graph.edge(id_name, hw_id, penwidth='2', tailport=default_tailport, headport=default_headport,
                  color=str(color))



def add_implementation(graph, id_name, color, secondary_color, backends, description=None, caveat=None, backend_caveats={}):
  if description == None:
    description = id_name


  graph.node(id_name, description, shape='box', color=str(color), width=default_node_width,
              style='rounded,filled', fontcolor = str(determine_font_color(color)))

  for b in backends:
    if b in backend_caveats:
      graph.edge(id_name, b, penwidth='2', headport=default_headport, tailport=default_tailport,
                 color=str(secondary_color), style="dashed", headlabel=add_footnote(str(backend_caveats[b])),
                 labelfloat="true", labeldistance="2.0",  weight="100")
    else:
      graph.edge(id_name, b, penwidth='2', headport=default_headport, tailport=default_tailport,
                 color=str(secondary_color),
                 weight="100")

  if caveat == None:
    graph.edge("root", id_name, penwidth='2', headport=default_headport, tailport=default_tailport,
               color=str(secondary_color))
  else:
    graph.edge("root", id_name, penwidth='2', headport=default_headport, tailport=default_tailport,
               color=str(secondary_color), style="dashed", labelfloat="true", labeldistance="2.0",
               headlabel=add_footnote(str(caveat)))


if __name__ == '__main__':
  graph = Digraph(node_attr={'shape': 'record', 'height': '.9'}, engine='dot', format=outputformat)


  ranksep=1.2
  if display_mode == "no-hardware-nodes":
    ranksep = 0.8

  rankdir = "TB"
  if layout == "left-right":
    rankdir = "LR"

  if outputformat != "svg":
    graph.attr('graph', mclimit='1000', ranksep=str(ranksep), dpi="250", rankdir=rankdir)
  else:
    graph.attr('graph', mclimit='1000', ranksep=str(ranksep), rankdir=rankdir)

  root_color = make_html_color("d5e8d4")
  graph.node("root", "SYCL source code", shape='box', color=str(root_color),
              style='rounded,filled', fontcolor = str(determine_font_color(root_color)))

  intel_blue          = make_html_color("127bca")
  acpp_grey        = make_html_color("555555")
  acpp_red         = make_html_color("c50d29")
  codeplay_purple     = make_html_color("993697")
  codeplay_light_blue = make_html_color("a7d5fd")
  xilinx_dark = make_html_color("171c2d")
  xilinx_red  = make_html_color("ee3a25")
  openmp_green = make_html_color("00737D")
  nvidia_green = make_html_color("76b900")
  amd_red      = make_html_color("ed1c24")
  arm_blue     = make_html_color("0091bd")
  renesas_blue = make_html_color("2a289d")
  cpu_color = make_html_color("a0a0a0")
  sycl_gtx_dark  = make_html_color("10739e")
  sycl_gtx_light = make_html_color("ff9b43")
  nec_blue = make_html_color("1414a8")

  #add_backend(graph, "generic_openmp", openmp_green, {'Any CPU' : cpu_color},
  #            description="OpenMP")

  ####################### DPC++ ##############################

  add_backend(graph, "dpcpp_host", intel_blue, {'Any CPU' : cpu_color},
              description="Host")

  add_backend(graph, "dpcpp_opencl", intel_blue, {
    'Intel CPUs' : intel_blue,
    'Intel GPUs' : intel_blue,
    'Intel FGPAs' : intel_blue
  }, description="OpenCL+SPIR-V")

  add_backend(graph, "dpcpp_cuda", intel_blue, {
    'NVIDIA GPUs' : nvidia_green
  }, description="CUDA\n(driver API)")

  add_backend(graph, "dpcpp_l0", intel_blue, {'Intel GPUs' : intel_blue},
              description="Level Zero")

  add_implementation(graph, "DPC++", intel_blue, intel_blue,
                     ["dpcpp_host", "dpcpp_opencl", "dpcpp_cuda", "dpcpp_l0"])

  ####################### AdaptiveCpp ##############################

  add_backend(graph, "acpp_cuda", acpp_grey, {'NVIDIA GPUs' : nvidia_green},
              description="CUDA\n(runtime API)")

  add_backend(graph, "acpp_rocm", acpp_grey, {'AMD GPUs' : amd_red},
              description="ROCm")

  add_backend(graph, "acpp_openmp", acpp_grey,
              {'Any CPU' : cpu_color}, description="OpenMP")

  add_backend(graph, "acpp_l0", acpp_grey, {'Intel GPUs' : intel_blue},
              description="Level Zero")

  add_backend(graph, "acpp_ocl", acpp_grey, {'OpenCL SPIR-V devices' : intel_blue},
              description="OpenCL")

  add_implementation(graph, "AdaptiveCpp", acpp_grey, acpp_red,
                     ["acpp_cuda", "acpp_rocm", "acpp_openmp", "acpp_l0", "acpp_ocl"])

  ####################### triSYCL ##############################

  add_backend(graph, "trisycl_opencl", xilinx_dark, {
    'Xilinx FPGAs' : xilinx_dark,
    'pocl' : make_html_color("333333"),
  }, description="OpenCL+SPIR-df")

  add_backend(graph, "trisycl_tbb", xilinx_dark, {'Any CPU' : cpu_color},
              description="TBB")


  add_backend(graph, "trisycl_openmp", xilinx_dark, {'Any CPU' : cpu_color}, description="OpenMP")

  add_implementation(graph, "triSYCL", xilinx_dark, xilinx_red,
                     ["trisycl_openmp", "trisycl_tbb", "trisycl_opencl"],
                     backend_caveats={'trisycl_opencl' : 'experimental'})

  ####################### neoSYCL ##############################

  add_backend(graph, "neosycl_veo", nec_blue, {'NEC SX-Aurora TSUBASA' : nec_blue}, description="VEO")
  add_implementation(graph, "neoSYCL", nec_blue, nec_blue, ["neosycl_veo"])

  ####################### ComputeCpp ##############################

  add_backend(graph, "computecpp_host", codeplay_purple, {'Any CPU' : cpu_color},
              description="Host")

  add_backend(graph, "computecpp_ptx", codeplay_purple, {
    'NVIDIA GPUs' : nvidia_green
  }, description="OpenCL+PTX")

  add_backend(graph, "computecpp_opencl", codeplay_purple, {
    'Intel CPUs' : intel_blue,
    'Intel GPUs' : intel_blue,
    'AMD GPUs' : amd_red,
    'ARM Mali' : arm_blue,
    'Renesas R-Car' : renesas_blue
  }, device_caveats = {'AMD GPUs' : "Only some drivers"},
    description="OpenCL+SPIR/SPIR-V")

  add_implementation(graph, "ComputeCpp", codeplay_purple, codeplay_light_blue,
                     backends=["computecpp_host", "computecpp_opencl", "computecpp_ptx"],
                     backend_caveats={'computecpp_ptx' : 'experimental'})

  ####################### sycl-gtx ##############################

  add_backend(graph, "sycl_gtx_opencl", sycl_gtx_dark, {
    'Any OpenCL 1.2 device' : sycl_gtx_dark
  }, description="OpenCL 1.2")
  add_implementation(graph, "sycl-gtx", sycl_gtx_dark, sycl_gtx_light, ["sycl_gtx_opencl"],
                     caveat='Non-standard macros required')

  graph.node("footnotes", footnote_text, shape="plain", style='filled', color="#ffffff00")
  graph.render('sycl-implementations')
