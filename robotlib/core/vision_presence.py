from __future__ import annotations


def choose_vision_event_response(
    category: str,
    count: int,
    *,
    responses_loader,
    chooser,
) -> str | None:
    responses = responses_loader()
    options = list(responses.get(category, []))
    if not options:
        return None
    template = chooser(options)
    try:
        return str(template).format(count=count)
    except Exception:
        return str(template)


def handle_vision_tick(
    camera_runtime: dict,
    detections: list[dict],
    *,
    choose_response,
    set_gesture,
    is_audio_playing,
    is_tts_active,
) -> str | None:
    current_count = len(detections)
    previous_count = int(camera_runtime.get("vision_last_detection_count", 0))
    camera_runtime["vision_last_detection_count"] = current_count

    if current_count == previous_count:
        return None
    if previous_count == 0 and current_count > 0:
        set_gesture(camera_runtime, "join")
        if bool(camera_runtime.get("suppress_next_join_after_interrupt", False)):
            camera_runtime["suppress_next_join_after_interrupt"] = False
            return None
        return choose_response("first_person_joined", current_count)
    if current_count > previous_count:
        set_gesture(camera_runtime, "join")
        return choose_response("more_people_joined", current_count)
    if current_count == 0:
        set_gesture(camera_runtime, "leave")
        if is_audio_playing() or is_tts_active():
            camera_runtime["suppress_next_join_after_interrupt"] = True
            return "__INTERRUPT_AUDIO__:me cayo"
        return choose_response("alone_again", current_count)
    set_gesture(camera_runtime, "leave")
    return choose_response("fewer_people_left", current_count)
