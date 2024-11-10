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
} from "react-leaflet";
import { Map } from "leaflet";
import "leaflet/dist/leaflet.css";
import dynamic from "next/dynamic";

export default function OpenStreetMapComponent({
  setRef,
}: {
  setRef: (ref: Map | null) => void;
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
      </MapContainer>
    </div>
  );
}
