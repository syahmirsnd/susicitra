"use strict";

/* ::::::::::::::::::::
:: GLobal Javascript ::
::::::::::::::::::::::: */

// ==== Sticky Menu ====
window.addEventListener("scroll", function () {
  let header = document.querySelector("header");
  header.classList.toggle("scrolling", window.scrollY > 0);
});

// ::::: GLobal Javascript ::::
// ================================Animate Interaction on Scroll ==================================
JOS.init({
  // disable: false, // Disable JOS gloabaly | Values :  'true', 'false'
  // debugMode: true, // Enable JOS debug mode | Values :  'true', 'false'
  passive: false, // Set the passive option for the scroll event listener | Values :  'true', 'false'

  once: true, // Disable JOS after first animation | Values :  'true', 'false' || Int : 0-1000
  animation: "fade-up", // JOS global animation type | Values :  'fade', 'slide', 'zoom', 'flip', 'fade-right', 'fade-left', 'fade-up', 'fade-down', 'zoom-in-right', 'zoom-in-left', 'zoom-in-up', 'zoom-in-down', 'zoom-out-right', 'zoom-out-left', 'zoom-out-up', 'zoom-out-down', 'flip-right', 'flip-left', 'flip-up', 'flip-down, spin, revolve, stretch, "my-custom-animation"
  // animationInverse: "static", // Set the animation type for the element when it is scrolled out of view | Values :  'fade', 'slide', 'zoom', 'flip', 'fade-right', 'fade-left', 'fade-up', 'fade-down', 'zoom-in-right', 'zoom-in-left', 'zoom-in-up', 'zoom-in-down', 'zoom-out-right', 'zoom-out-left', 'zoom-out-up', 'zoom-out-down', 'flip-right', 'flip-left', 'flip-up', 'flip-down, spin, revolve, stretch, "my-custom-animation"
  timingFunction: "ease", // JOS global timing function | Values :  'ease', 'ease-in', 'ease-out', 'ease-in-out', 'linear', 'step-start', 'step-end', 'steps()', 'cubic-bezier()', 'my-custom-timing-function'
  //mirror : false, // Set whether the element should animate back when scrolled out of view | Values :  'true', 'false'
  threshold: 0, // Set gloabal the threshold for the element to be visible | Values :  0-1
  delay: 0.5, // Set global the delay for the animation to start | Values :  0,1,2,3,4,5
  duration: 0.7, // Set global the duration for the animation playback | Values :  flota : 0-1 & int : 0,1,2,3,4,5

  // startVisible: "true", // Set whether the element should animate when the page is loaded | Values :  'true', 'false' || MS : 0-10000
  scrollDirection: "down", // Set the scroll direction for the element to be visible | Values :  'up', 'down', 'none'
  //scrollProgressDisable: true // disable or enable scroll callback function | Values :  'true', 'false'
  // intersectionRatio: 0.4, // Set the intersection ratio between which the element should be visible | Values :  0-1 (automaticaly set)
  // rootMargin_top: "0%", // Set by which percent the element should animate out (Recommended value between 10% to -30%)
  // rootMargin_bottom: "-50%", // Set by which percent the element should animate out (Recommended value between -10% to -60%)
  rootMargin: "0% 0% 15% 0%", // Set the root margin for the element to be visible | Values :  _% _% _% _%  (automaticaly set)
});

// ======================================== Accordion ======================================
let accordions = document.querySelectorAll(".accordion-item");
accordions.forEach((item) => {
  let label = item.querySelector(".accordion-header");
  label.addEventListener("click", () => {
    accordions.forEach((accordionItem) => {
      accordionItem.classList.remove("active");
    });
    item.classList.toggle("active");
  });
});

//=========== Progress Bar =============
// Find the element with the class 'progress-bar'
const progressBars = document.querySelectorAll(".progress-bar");

// Iterate over each progress bar element
progressBars.forEach((progressBar) => {
  // Get the value of the 'data-percentage-value' attribute for each progress bar
  const percentageValue = progressBar.getAttribute("data-percentage-value");
  progressBar.style.width = percentageValue + "%";
});

//=========== Pricing Button =============
function toggleSwitch() {
  var month = document.querySelectorAll(".month");
  var annual = document.querySelectorAll(".annual");
  for (var i = 0; i < month.length; i++) {
    if (document.getElementById("toggle").checked == true) {
      month[i].classList.add("hidden");
      annual[i].classList.remove("hidden");
    } else {
      month[i].classList.remove("hidden");
      annual[i].classList.add("hidden");
    }
  }
}

// =========== Show Image On Hover =============
function showImage(event) {
  const hoverOnItem = event.currentTarget;
  const hoverOnImage = hoverOnItem.querySelector(".hover-on-image");
  const hoveredImage = document.getElementById("hoveredImage");

  // Set the hovered image source to the hoverOn image source
  hoveredImage.src = hoverOnImage.src;

  // Show the hovered image
  hoveredImage.style.display = "block";

  // Move the hovered image with the cursor
  document.addEventListener("mousemove", moveImage);

  function moveImage(event) {
    const x = event.clientX;
    const y = event.clientY;

    hoveredImage.style.transform = `translate(${x}px, ${y}px)`;
  }

  // Hide the hovered image when mouse leaves the hoverOn item
  hoverOnItem.onmouseleave = () => {
    hoveredImage.style.display = "none";
    document.removeEventListener("mousemove", moveImage);
  };
}

/* ::::::::::::::::::::::
:: Template Javascript ::
::::::::::::::::::::::::: */

// =========== Project Slider - 1 =============
const projectSliderOne = new Swiper(".projectSliderOne", {
  slidesPerView: 1,
  pagination: {
    el: ".swiper-pagination",
    type: "progressbar",
  },
  navigation: {
    nextEl: ".project-button-next",
    prevEl: ".project-button-prev",
  },
  spaceBetween: 24,
  breakpoints: {
    768: {
      slidesPerView: 2,
    },
    1200: {
      slidesPerView: 3,
    },
    1600: {
      slidesPerView: 4,
    },
  },
});

// =========== Project Slider - 2 =============
const projectSliderTwo = new Swiper(".projectSliderTwo", {
  slidesPerView: 1,
  spaceBetween: 24,
  breakpoints: {
    768: {
      slidesPerView: 2,
    },
    1200: {
      slidesPerView: 3,
    },
  },
});

// =========== Project Slider - 3 =============
const projectSliderThree = new Swiper(".projectSliderThree", {
  slidesPerView: 1,
  spaceBetween: 24,
  scrollbar: {
    el: ".swiper-scrollbar",
    draggable: true,
  },
  breakpoints: {
    640: {
      slidesPerView: 1,
    },
    900: {
      slidesPerView: 2,
    },
    1600: {
      slidesPerView: 3,
    },
  },
});

// =========== Project Slider - 4 =============
const projectSliderFour = new Swiper(".projectSliderFour", {
  pagination: {
    el: ".swiper-pagination",
    type: "fraction",
  },
  spaceBetween: 24,
  loop: true,
  breakpoints: {
    640: {
      slidesPerView: 1,
    },
    900: {
      slidesPerView: 2,
    },
    1600: {
      slidesPerView: 8,
    },
  },
});

// =========== Client Logo Slider =============
const clientSlider = new Swiper(".client-slider", {
  slidesPerView: 1,
  loop: true,
  speed: 5000,
  autoplay: {
    delay: 1,
    disableOnInteraction: false,
  },
  breakpoints: {
    576: {
      slidesPerView: 2,
    },
    768: {
      slidesPerView: 3,
    },
    992: {
      slidesPerView: 4,
    },
    1200: {
      slidesPerView: 5,
    },
    1400: {
      slidesPerView: 6,
    },
  },
});

const clientSliderTwo = new Swiper(".client-slider-2", {
  slidesPerView: 1,
  loop: true,
  speed: 5000,
  autoplay: {
    delay: 1,
    disableOnInteraction: false,
  },
  breakpoints: {
    576: {
      slidesPerView: 2,
    },
    992: {
      slidesPerView: 3,
    },
    1200: {
      slidesPerView: 4,
    },
  },
});
const clientSliderThree = new Swiper(".client-slider-3", {
  slidesPerView: 1,
  loop: true,
  speed: 5000,
  autoplay: {
    delay: 1,
    disableOnInteraction: false,
    reverseDirection: true,
  },
  breakpoints: {
    576: {
      slidesPerView: 2,
    },
    992: {
      slidesPerView: 3,
    },
    1200: {
      slidesPerView: 4,
    },
  },
});

// =========== Instagram Slider =============
const instagramSlider = new Swiper(".instagram-slider", {
  // Optional parameters
  slidesPerView: 1,

  breakpoints: {
    576: {
      slidesPerView: 2,
    },
    768: {
      slidesPerView: 2,
    },
    992: {
      slidesPerView: 3,
    },
    1200: {
      slidesPerView: 4,
    },
    1400: {
      slidesPerView: 5,
    },
  },
});

// =========== Testimonial Slider - 1 =============
const testimonialSliderOne = new Swiper(".testimonial-slider-1", {
  // Optional parameters
  loop: true,

  // If we need pagination
  pagination: {
    el: ".swiper-pagination",
    clickable: true,
  },
});

const testimonialSlideTwo = new Swiper(".testimonial-slider-2", {
  // Optional parameters
  loop: true,
});

const testimonialSlideThree = new Swiper(".testimonial-slider-3", {
  // Optional parameters
  loop: true,
  slidesPerView: 1,
  spaceBetween: 24,
  breakpoints: {
    768: {
      slidesPerView: 2,
    },
  },
});

// =========== Service Slider - 1 =============
const serviceSliderOne = new Swiper(".service-slider-one", {
  slidesPerView: 1,
  loop: true,
  spaceBetween: 30,
  // Navigation arrows
  navigation: {
    nextEl: ".slider-button-next",
    prevEl: ".slider-button-prev",
  },
  breakpoints: {
    768: {
      slidesPerView: 1.5,
    },
    1400: {
      slidesPerView: 2.5,
    },
  },
});

// =========== Home-9: Hero Slide =============
const hero9Slider = new Swiper(".hero-9-slider", {
  loop: true,
  speed: 5000,
  spaceBetween: 24,
  slidesPerView: 2,
  autoplay: {
    delay: 1,
    disableOnInteraction: false,
  },
  breakpoints: {
    576: {
      slidesPerView: 3,
    },
  },
});
const hero9SliderReverse = new Swiper(".hero-9-slider-reverse", {
  loop: true,
  speed: 5000,
  spaceBetween: 24,
  slidesPerView: 2,
  autoplay: {
    delay: 1,
    disableOnInteraction: false,
    reverseDirection: true,
  },
  breakpoints: {
    576: {
      slidesPerView: 3,
    },
  },
});
