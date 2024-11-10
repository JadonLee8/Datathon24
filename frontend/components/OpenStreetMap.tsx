import React, {
  useState,
  useRef,
  LegacyRef,
  MutableRefObject,
  useEffect,
  useCallback,
} from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  ZoomControl,
  ImageOverlay,
  ImageOverlayProps,
} from "react-leaflet";
import { Map } from "leaflet";
import "leaflet/dist/leaflet.css";
import dynamic from "next/dynamic";

// interface ImageOverlayProps {
//   bounds: [[number, number], [number, number]];
//   url: string;
// }
export default function OpenStreetMapComponent({
  setRef,
  imageOverlay,
}: {
  setRef: (ref: Map | null) => void;
  imageOverlay: ImageOverlayProps | null;
}) {
  return (
    <div
      className="bg-lightmaroon ring-2 ring-white ring-offset-lightmaroon
    ring-offset-2 overflow-hidden rounded-sm"
    >
      <MapContainer
        center={{
          lat: 38.719805,
          lng: -457.717365,
        }}
        zoom={5}
        ref={setRef}
        className="h-screen"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />
        {imageOverlay && (
          <ImageOverlay bounds={imageOverlay.bounds} url={imageOverlay.url} />
        )}
      </MapContainer>
    </div>
  );
}
