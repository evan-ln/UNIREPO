library(rjson)

#' Normalise vectors
#'
#' We use a number of vector normalisation strategies in Computational
#' Musicology. This function brings them together into one place, along with
#' common alternative names.
#'
#' The following methods are supported.
#' \describe{
#'     \item{\code{identity},\code{id}}{No normalisation.}
#'     \item{\code{harmonic}}{Harmonic mean.}
#'     \item{\code{manhattan},\code{L1}}{Manhattan (L1) norm.}
#'     \item{\code{euclidean},\code{L2}}{Euclidean (L2) norm.}
#'     \item{\code{chebyshev},\code{maximum}}{Chebyshev (maximum) norm.}
#'     \item{\code{aitchison},\code{clr}}{Aitchison's clr transformation.}
#'     \item{\code{softmax}}{Softmax.}}
#'
#' @param v A numeric vector.
#' @param method A character string indicating which normalization to use (see
#'   Details). Default is the Euclidean norm.
#'
#' @importFrom magrittr %>%
#' @export
#'
#' @examples
#' library(tidyverse)
#' get_tidy_audio_analysis("6IQILcYkN2S2eSu5IHoPEH") %>%
#'   select(segments) %>%
#'   unnest(segments) %>%
#'   mutate(pitches = map(pitches, compmus_normalise, "euclidean"))
compmus_normalise <- function(v, method = "euclidean") {
  ## Supported functions

  harmonic <- function(v) v * sum(1 / abs(v))
  manhattan <- function(v) v / sum(abs(v))
  euclidean <- function(v) v / sqrt(sum(v^2))
  chebyshev <- function(v) v / max(abs(v))
  clr <- function(v) {
    lv <- log(v)
    lv - mean(lv)
  }
  softmax <- function(v) {
    exp(v) / sum(exp(v))
  }

  ## Method aliases

  METHODS <-
    list(
      identity = identity,
      harmonic = harmonic,
      manhattan = manhattan,
      L1 = manhattan,
      euclidean = euclidean,
      L2 = euclidean,
      chebyshev = chebyshev,
      maximum = chebyshev,
      aitchison = clr,
      clr = clr,
      softmax = softmax
    )

  ## Function selection

  if (!is.na(i <- pmatch(method, names(METHODS)))) {
    METHODS[[i]](v)
  } else {
    stop("The method name is ambiguous or the method is unsupported.")
  }
}

#' @describeIn compmus_normalise Normalize vectors
#' @export
compmus_normalize <- compmus_normalise

#' Pairwise distances in long format
#'
#' We use a number of distance measures in Computational Musicology.
#' \code{compmus_long_distance} brings them together into one place, along with
#' common alternative names. In order to support plotting, the distances are
#' returned in long format rather than matrix format. It is designed for
#' convenience, not speed.
#'
#' The following methods are supported. \describe{
#' \item{\code{manhattan},\code{citybolock},\code{taxicab},\code{L1},\code{totvar}}{Manhattan
#' distance.} \item{\code{euclidean},\code{L2}}{Euclidean distance.}
#' \item{\code{chebyshev},\code{maximum}}{Chebyshev distance.}
#' \item{\code{pearson},\code{correlation}}{Pearson's pseudo-distance.}
#' \item{\code{cosine}}{Cosine pseudo-distance.} \item{\code{angular}}{Angular
#' distance.} \item{\code{aitchison}}{Aitchison distance.} }
#'
#' @param xdat,ydat,dat Data frames with \code{start} and \code{duration}
#'   columns.
#' @param feature An (unquoted) column name over which to compute distances.
#' @param method A character string indicating which distance metric to use (see
#'   Details). Default is Euclidean distance.
#' @return A tibble with columns \code{xstart}, \code{xduration}, \code{ystart},
#'   \code{yduration}, and \code{d}.
#'
#' @importFrom stats cor
#' @importFrom magrittr %>%
#' @importFrom rlang !!
#' @export
#'
#' @examples
#' library(tidyverse)
#' tallis <-
#'   get_tidy_audio_analysis("2J3Mmybwue0jyQ0UVMYurH") %>%
#'   select(segments) %>%
#'   unnest(segments) %>%
#'   mutate(pitches = map(pitches, compmus_normalise, "manhattan"))
#' chapelle <-
#'   get_tidy_audio_analysis("4ccw2IcnFt1Jv9LqQCOYDi") %>%
#'   select(segments) %>%
#'   unnest(segments) %>%
#'   mutate(pitches = map(pitches, compmus_normalise, "manhattan"))
#'
#' compmus_long_distance(tallis, chapelle, pitches, method = "euclidean")
#'
#' compmus_self_similarity(tallis, pitches, method = "aitchison")
compmus_long_distance <- function(xdat, ydat, feature, distance = "euclidean") {
  feature <- enquo(feature)

  ## Supported functions

  manhattan <- function(x, y) sum(abs(x - y))
  euclidean <- function(x, y) sqrt(sum((x - y)^2))
  chebyshev <- function(x, y) max(abs(x - y))
  pearson <- function(x, y) 1 - cor(x, y)
  cosine <- function(x, y) {
    1 - sum(compmus_normalise(x, "euc") * compmus_normalise(y, "euc"))
  }
  angular <- function(x, y) 2 * acos(1 - cosine(x, y)) / pi
  aitchison <- function(x, y) {
    euclidean(compmus_normalise(x, "clr"), compmus_normalise(y, "clr"))
  }

  ## Method aliases

  METHODS <-
    list(
      manhattan = manhattan,
      cityblock = manhattan,
      taxicab = manhattan,
      L1 = manhattan,
      totvar = manhattan,
      euclidean = euclidean,
      L2 = euclidean,
      chebyshev = chebyshev,
      maximum = chebyshev,
      pearson = pearson,
      correlation = pearson,
      cosine = cosine,
      angular = angular,
      aitchison = aitchison
    )

  ## Function selection

  if (!is.na(i <- pmatch(distance, names(METHODS)))) {
    dplyr::cross_join(
      xdat |>
        dplyr::rename(xtime = time) |>
        tidyr::nest(.by = xtime) |>
        dplyr::transmute(
          xtime,
          x =
            purrr::map(
              data,
              \(df) df |> dplyr::arrange(!!feature) |> dplyr::pull(value)
            )
        ) |>
        dplyr::arrange(xtime) |>
        dplyr::filter(dplyr::row_number(xtime) %% 10 == 5),
      ydat |>
        dplyr::rename(ytime = time) |>
        tidyr::nest(.by = ytime) |>
        dplyr::transmute(
          ytime,
          y =
            purrr::map(
              data,
              \(df) df |> dplyr::arrange(!!feature) |> dplyr::pull(value)
            )
        ) |>
        dplyr::arrange(ytime) |>
        dplyr::filter(dplyr::row_number(ytime) %% 10 == 5),
    ) |>
      dplyr::transmute(
        xtime,
        ytime,
        d = purrr::map2_dbl(x, y, METHODS[[i]])
      )
  } else {
    stop("The distance name is ambiguous or the method is unsupported.")
  }
}

#' @describeIn compmus_long_distance Self-similarity matrices in long format
#' @importFrom rlang !! enquo
#' @export
compmus_self_similarity <- function(dat, feature, distance = "euclidean") {
  feature <- enquo(feature)
  compmus_long_distance(dat, dat, !!feature, distance)
}

#' Match chroma vectors against templates
#'
#' Compares chroma vectors in a data frame against a list of templates, most
#' likely key or chord profiles.
#'
#' @param dat A data frame containing chroma vectors in a \code{pitches} column.
#' @param templates A data frame with a \code{name} column for each template and
#'   the templates themselves in a \code{template} column.
#' @param method A character string indicating which distance metric to use (see
#'   \code{\link{compmus_long_distance}}). Default is cosine distance.
#' @param norm An optional character string indicating the method for
#'   pre-normalising each vector with \code{\link{compmus_normalise}}. Default
#'   is Euclidean.

#' @return A tibble with columns \code{start}, \code{duration}, \code{name}, and
#'   \code{d}.
#'
#' @importFrom magrittr %>%
#' @export
#'
#' @examples
#' library(tidyverse)
#' circshift <- function(v, n) {
#'   if (n == 0) v else c(tail(v, n), head(v, -n))
#' }
#' major_chord <-
#'   c(1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0)
#' minor_chord <-
#'   c(1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0)
#' chord_templates <-
#'   tribble(
#'     ~name, ~template,
#'     "D:min", circshift(minor_chord, 2),
#'     "F:maj", circshift(major_chord, 5),
#'     "A:min", circshift(minor_chord, 9),
#'     "C:maj", circshift(major_chord, 0),
#'     "E:min", circshift(minor_chord, 4),
#'     "G:maj", circshift(major_chord, 7),
#'     "B:min", circshift(minor_chord, 11)
#'   )
#'
#' get_tidy_audio_analysis("5UVsbUV0Kh033cqsZ5sLQi") %>%
#'   compmus_align(sections, segments) %>%
#'   select(sections) %>%
#'   unnest(sections) %>%
#'   mutate(
#'     pitches =
#'       map(segments,
#'         compmus_summarise, pitches,
#'         method = "mean", norm = "manhattan"
#'       )
#'   ) %>%
#'   compmus_match_pitch_template(chord_templates, "euclidean", "manhattan")
compmus_match_pitch_templates <- function(dat, templates, norm = "euclidean", distance = "cosine") {

    ## Supported functions

  manhattan <- function(x, y) sum(abs(x - y))
  euclidean <- function(x, y) sqrt(sum((x - y)^2))
  chebyshev <- function(x, y) max(abs(x - y))
  pearson <- function(x, y) 1 - cor(x, y)
  cosine <- function(x, y) {
    1 - sum(compmus_normalise(x, "euc") * compmus_normalise(y, "euc"))
  }
  angular <- function(x, y) 2 * acos(1 - cosine(x, y)) / pi
  aitchison <- function(x, y) {
    euclidean(compmus_normalise(x, "clr"), compmus_normalise(y, "clr"))
  }

  ## Method aliases

  METHODS <-
    list(
      manhattan = manhattan,
      cityblock = manhattan,
      taxicab = manhattan,
      L1 = manhattan,
      totvar = manhattan,
      euclidean = euclidean,
      L2 = euclidean,
      chebyshev = chebyshev,
      maximum = chebyshev,
      pearson = pearson,
      correlation = pearson,
      cosine = cosine,
      angular = angular,
      aitchison = aitchison
    )

  ## Function selection

  if (!is.na(i <- pmatch(distance, names(METHODS)))) {
    dplyr::cross_join(
      dat |>
        tidyr::nest(.by = time) |>
        dplyr::transmute(
          time,
          x =
            purrr::map(
              data,
              \(df) df |> dplyr::arrange(pc) |> dplyr::pull(value)
            )
        ) |>
        dplyr::arrange(time) |>
        dplyr::filter(dplyr::row_number(time) %% 10 == 5),
      templates |>
        dplyr::transmute(
          name,
          y = purrr::map(template, \(v) compmus_normalise(v, norm))
        )
      ) |>
      dplyr::transmute(
        time,
        name = factor(name, levels = purrr::pluck(templates, "name")),
        d = purrr::map2_dbl(x, y, METHODS[[i]])
      )
  } else {
    stop("The distance name is ambiguous or the method is unsupported.")
  }
}

circshift <- function(v, n) {
  if (n == 0) v else c(utils::tail(v, n), utils::head(v, -n))
}

compmus_chroma <- function(file, norm = "identity") {
  rjson::fromJSON(file = file) |>
    purrr::pluck("tonal", "hpcp") |>
    purrr::map(
      \(v) {
        v |>
          circshift(-8) |>
          matrix(3, 12) |>
          colMeans() |>
          compmus_normalise(method = norm) |>
          tibble::enframe(name = "pc") |>
          dplyr::mutate(pc = pc - 1)
      }
    ) |>
    purrr::list_rbind(names_to = "time") |>
    dplyr::mutate(time = 2048 * time / 44100)
}

compmus_mfccs <- function(file, norm = "identity") {
  rjson::fromJSON(file = file) |>
    purrr::pluck("lowlevel", "mfcc") |>
    purrr::map(
      \(v) {
        v |>
          compmus_normalise(method = norm) |>
          tibble::enframe(name = "mfcc") |>
          dplyr::mutate(mfcc = mfcc - 1)
      }
    ) |>
    purrr::list_rbind(names_to = "time") |>
    dplyr::mutate(time = 1024 * time / 44100)
}

compmus_energy_novelty <- function(file) {
  rjson::fromJSON(file = file) |>
    purrr::pluck("lowlevel", "loudness_ebu128", "short_term") |>
    (\(v) v[2:length(v)] - v[1:(length(v) - 1)])() |>
    pmax(0) |>
    (\(v) tibble::tibble(t = 0.1 * (1:length(v)) + 0.1, novelty = v))()
}

compmus_spectral_novelty <- function(file, norm = "euclidean", distance = "euclidean") {
  ## Supported functions

  manhattan <- function(x, y) sum(abs(x - y))
  euclidean <- function(x, y) sqrt(sum((x - y)^2))
  chebyshev <- function(x, y) max(abs(x - y))
  pearson <- function(x, y) 1 - cor(x, y)
  cosine <- function(x, y) {
    1 - sum(compmus_normalise(x, "euc") * compmus_normalise(y, "euc"))
  }
  angular <- function(x, y) 2 * acos(1 - cosine(x, y)) / pi
  aitchison <- function(x, y) {
    euclidean(compmus_normalise(x, "clr"), compmus_normalise(y, "clr"))
  }

  ## Method aliases

  METHODS <-
    list(
      manhattan = manhattan,
      cityblock = manhattan,
      taxicab = manhattan,
      L1 = manhattan,
      totvar = manhattan,
      euclidean = euclidean,
      L2 = euclidean,
      chebyshev = chebyshev,
      maximum = chebyshev,
      pearson = pearson,
      correlation = pearson,
      cosine = cosine,
      angular = angular,
      aitchison = aitchison
    )

  ## Function selection

  if (!is.na(i <- pmatch(distance, names(METHODS)))) {
    rjson::fromJSON(file = file) |>
      purrr::pluck("lowlevel", "mfcc") |>
      purrr::map(\(v) compmus_normalise(v, norm)) |>
      (\(v) purrr::map2_dbl(v[2:length(v)], v[1:(length(v) - 1)], METHODS[[i]]))() |>
      (\(v) tibble::tibble(t = 1024 * (1:length(v) - 0.5) / 44100, novelty = v))()
  } else {
    stop("The distance name is ambiguous or the method is unsupported.")
  }
}

#' @importFrom magrittr %>%
.sample_tempogram <- function(y, f_s, window_size, hop_size, window_function, cyclic, bpms) {
  window <- window_function(window_size)
  if (cyclic) {
    bpm_octaves <- rep(bpms, 5)
    bpms <- bpms %>%
      tcrossprod(2^(-2:2)) %>%
      as.vector()
  } else {
    bpm_octaves <- bpms
  }
  bases <-
    exp(tcrossprod(-2 * pi * 1i * (bpms / 60) / f_s, 0:(window_size - 1)))
  if (length(y) > window_size) {
    starts <- seq(1, length(y) - window_size, by = hop_size)
  } else {
    starts <- 1
  }
  windowed <- matrix(0, window_size, length(starts))
  for (n in 1:length(starts)) {
    windowed[, n] <- window * y[starts[n]:(starts[n] + window_size - 1)]
  }
  (bases %*% windowed) %>%
    abs() %>%
    magrittr::raise_to_power(2) %>%
    as.vector() %>%
    tibble::tibble(
      time = rep((starts + window_size / 2) / f_s, each = length(bpms)),
      bpm = rep(bpm_octaves, times = length(starts)),
      power = .
    ) %>%
    dplyr::group_by(time, bpm) %>%
    dplyr::summarise(power = sum(power)) %>%
    dplyr::mutate(power = power / max(power)) %>%
    dplyr::ungroup()
}

#' Compute a tempogram from Essentia segment onsets
#'
#' Computes a Fourier-based tempogram based on onsets of Spotify segments.
#' Returns a tibble with \code{time}, \code{bpm}, and \code{power} columns.
#' Power is normalised to a max of 1 (Chebyshev norm) within each time point.
#'
#' @param track_analysis Spotify audio analysis as returned by
#'   \code{\link{get_tidy_audio_analysis}}.
#' @param window_size Window size in seconds (default 8).
#' @param hop_size Hop size in seconds (default 1).
#' @param cyclic Boolean stating whether the tempogram should be cyclic (default
#'   not).
#' @param bpms Vector of tempi in beats per minute to include in the tempogram
#'   (default 30--200 for non-cyclic and 80--160 for cyclic, inclusive of all
#'   integer tempi).
#' @param window_function Window function for the Fourier analysis (default
#'   Hamming).
#'
#' @importFrom magrittr %>%
#' @export
#'
#' @examples
#' library(tidyverse)
#' get_tidy_audio_analysis("6PJasPKAzNLSOzxeAH33j2") %>%
#'   tempogram(window_size = 4, hop_size = 2)
compmus_tempogram <- function(file, window_size = 8, hop_size = 1, cyclic = FALSE, bpms = if (cyclic) 80:160 else 30:600, window_function = signal::hamming) {
  onsets <-
    rjson::fromJSON(file = file) |>
    purrr::pluck("rhythm", "beats_position") |>
    magrittr::multiply_by(44100) %>%
    round()

  confidence <-
    rjson::fromJSON(file = file) |>
    purrr::pluck("rhythm", "beats_loudness")
  duration <-
    rjson::fromJSON(file = file) |>
    purrr::pluck("metadata", "audio_properties", "length")

  novelty <- rep(0, 44100 * duration)
  novelty[onsets + 1] <- confidence

  .sample_tempogram(
    novelty,
    f_s = 44100,
    window_size = 44100 * window_size,
    hop_size = 44100 * hop_size,
    cyclic = cyclic,
    bpms = bpms,
    window_function = window_function
  )
}
