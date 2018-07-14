#ifndef SYCU_INFO_PLATFORM_HPP
#define SYCU_INFO_PLATFORM_HPP



namespace cl {
namespace sycl {
namespace info {

enum class platform : unsigned int {
  /** Returns the profile name (as a string_class) supported by the
      implementation.

      Can be either FULL PROFILE or EMBEDDED PROFILE.
  */
  profile,

  /** Returns the OpenCL software driver version string in the form major
      number.minor number (as a string_class)
  */
  version,

  /** Returns the name of the platform (as a string_class)
  */
  name,

  /** Returns the string provided by the platform vendor (as a string_class)
  */
  vendor,

  /** Returns a space-separated list of extension names supported by the
      platform (as a string_class)
  */
  extensions


};


}
}
}


#endif
